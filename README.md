# LLMs


from google.colab import userdata
import pandas as pd
import numpy as np
import logging
import random
import threading
import sys
import time
import re
from typing import TypedDict, List

# Import LangGraph and related components
from langgraph.graph import END, StateGraph
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Setup logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------------------------------------------------
# Data Loading and Document Preparation
# --------------------------------------------------------------------
groq_api = userdata.get("groq_api_key")
df = pd.read_csv("/content/defects.csv")  # Contains defects and solutions

# Load the test cases CSV into a global DataFrame.
# We expect columns: "Error", "Module", "TestCase", "Type"
try:
    test_cases_df = pd.read_csv("/content/test_cases.csv")
except Exception as e:
    logging.warning("Test cases file not found or unreadable. Creating an empty DataFrame.")
    test_cases_df = pd.DataFrame(columns=["Error", "Module", "TestCase", "Type"])

docs = []
for _, row in df.iterrows():
    if pd.notna(row["Description"]) and pd.notna(row["Solution"]):
        docs.append(Document(
            page_content=row["Description"],
            metadata={
                "solution": row["Solution"],
                "module": row["Module"]
            }
        ))

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

# --------------------------------------------------------------------
# Helper: Classify Test Case (Positive/Negative) based on keywords
# --------------------------------------------------------------------
def classify_test_case(tc_text: str) -> str:
    negative_keywords = [
        "fail", "error", "misconfigured", "incorrect", "doesn't work",
        "not work", "negative", "invalid", "wrong", "missing", "unexpected",
        "should not", "incorrectly", "failure", "reject", "malformed",
        "timeout", "invalid input", "edge case", "out of bounds"
    ]
    text_lower = tc_text.lower()
    return "negative" if any(kw in text_lower for kw in negative_keywords) else "positive"

# --------------------------------------------------------------------
# Helper: Get CSV Test Cases Filtered by Module and Error
# --------------------------------------------------------------------
def get_csv_test_cases(module: str, error: str):
    """
    Returns a tuple of two lists: (positive_test_cases, negative_test_cases).
    Each test case is considered valid if it contains the required sections.
    The CSV is filtered by both the given module and error.
    """
    required_sections = [
        "Test Scenario",
        "Test Steps",
        "Pre Requisites",
        "Expected Results",
        "Pass/Fail Criteria"
    ]
    def is_valid_tc(tc_text: str) -> bool:
        return all(section in tc_text for section in required_sections)
    
    positive_cases = []
    negative_cases = []
    
    global test_cases_df
    df_filtered = test_cases_df.copy()
    if not df_filtered.empty:
        if "Module" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["Module"] == module]
        if "Error" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["Error"] == error]
    
        for _, row in df_filtered.iterrows():
            # Look for the test case text in the "TestCase" column.
            tc_text = row.get("TestCase", "")
            if pd.isna(tc_text):
                continue
            tc_text = str(tc_text).strip()
            if is_valid_tc(tc_text):
                classification = classify_test_case(tc_text)
                if classification == "positive":
                    positive_cases.append(tc_text)
                else:
                    negative_cases.append(tc_text)
    return positive_cases, negative_cases

# --------------------------------------------------------------------
# Helper: Save New Test Cases to CSV (avoiding duplicates)
# --------------------------------------------------------------------
def save_new_test_cases(new_cases: List[dict]):
    """
    new_cases: list of dictionaries with keys "Error", "Module", "TestCase", "Type".
    This function appends only new (non-duplicate) test cases to the global CSV.
    """
    global test_cases_df
    required_columns = ["Error", "Module", "TestCase", "Type"]
    # Create test_cases_df if empty
    if test_cases_df.empty:
        test_cases_df = pd.DataFrame(columns=required_columns)
    
    rows_to_add = []
    for case in new_cases:
        duplicate = test_cases_df[
            (test_cases_df["Error"] == case["Error"]) &
            (test_cases_df["Module"] == case["Module"]) &
            (test_cases_df["TestCase"] == case["TestCase"]) &
            (test_cases_df["Type"] == case["Type"])
        ]
        if duplicate.empty:
            rows_to_add.append(case)
    if rows_to_add:
        new_df = pd.DataFrame(rows_to_add)
        test_cases_df = pd.concat([test_cases_df, new_df], ignore_index=True)
        test_cases_df.to_csv("/content/test_cases.csv", index=False)
        logging.info("Saved %d new test case(s) to CSV.", len(rows_to_add))
    else:
        logging.info("No new test cases to save (duplicates skipped).")

# --------------------------------------------------------------------
# Define Agent State and LLM Initialization
# --------------------------------------------------------------------
class AgentState(TypedDict):
    input: str
    context: List[Document]
    response: str

llm = ChatGroq(
    groq_api_key=groq_api,
    temperature=0.3,
    model_name="gemma2-9b-it",
)

# --------------------------------------------------------------------
# Workflow Node: Validate or Generate Test Cases (with CSV integration and storage)
# --------------------------------------------------------------------
def validate_or_generate_test_cases(state: AgentState):
    try:
        if not state["context"]:
            return {"response": "**Error**: The defect could not be found in the database."}
        context = state["context"][0]
        error_message = state["input"]
        solution = context.metadata["solution"]
        module = context.metadata["module"]

        # Generate explanation for the solution.
        explanation_prompt = """
        [INST] Explain why this solution fixes the following error:
        Error: {error}
        Solution: {solution}
        [/INST]
        """
        explanation_template = ChatPromptTemplate.from_template(explanation_prompt)
        formatted_explanation = explanation_template.format_prompt(error=error_message, solution=solution)
        explanation = llm.invoke(formatted_explanation.to_messages()).content.strip()

        # Delimiter used by the LLM outputs.
        delimiter = "\n### END TEST CASE ###\n"
        required_count = 2  # We require 2 positive and 2 negative test cases.

        # Attempt to fetch test cases from the CSV file based on module and error.
        pos_csv, neg_csv = get_csv_test_cases(module, error_message)
        logging.info("Fetched %d positive and %d negative CSV test cases.", len(pos_csv), len(neg_csv))

        new_generated_cases = []  # List to store any newly generated test cases for saving.

        # If valid CSV test cases exist, and we have at least 2 of each, use them.
        if len(pos_csv) >= required_count and len(neg_csv) >= required_count:
            final_pos = pos_csv[:required_count]
            final_neg = neg_csv[:required_count]
        else:
            # Determine how many test cases are missing.
            missing_pos = max(0, required_count - len(pos_csv))
            missing_neg = max(0, required_count - len(neg_csv))
            generated_pos = []
            generated_neg = []
            
            if missing_pos > 0:
                pos_prompt = """
                [INST] Generate EXACTLY {count} POSITIVE test case(s) for:
                Error: {error}
                Solution: {solution}

                Each test case MUST include the following sections, and end with the delimiter "### END TEST CASE ###":
                - **Test Scenario**: A short description of the scenario.
                - **Test Steps**: Step-by-step instructions.
                - **Pre Requisites**: Conditions before running the test.
                - **Expected Results**: What should happen if the solution works.
                - **Pass/Fail Criteria**: How to determine if the test passes.

                Output format (including the delimiter):
                1. **Test Scenario**: 
                   **Test Steps**: 
                   **Pre Requisites**: 
                   **Expected Results**: 
                   **Pass/Fail Criteria**: 
                ### END TEST CASE ###
                """.format(count=missing_pos, error=error_message, solution=solution)
                pos_template = ChatPromptTemplate.from_template(pos_prompt)
                formatted_pos = pos_template.format_prompt().to_messages()
                pos_response = llm.invoke(formatted_pos).content.strip()
                generated_pos = [tc.strip() for tc in re.split(delimiter, pos_response) if tc.strip()]
                generated_pos = generated_pos[:missing_pos]
            
            if missing_neg > 0:
                neg_prompt = """
                [INST] Generate EXACTLY {count} NEGATIVE test case(s) for:
                Error: {error}
                Solution: {solution}

                Each test case MUST include the following sections, and end with the delimiter "### END TEST CASE ###":
                - **Test Scenario**: A short description of the scenario.
                - **Test Steps**: Step-by-step instructions.
                - **Pre Requisites**: Conditions before running the test.
                - **Expected Results**: What should happen if the solution fails.
                - **Pass/Fail Criteria**: How to determine if the test fails.

                Output format (including the delimiter):
                1. **Test Scenario**: 
                   **Test Steps**: 
                   **Pre Requisites**: 
                   **Expected Results**: 
                   **Pass/Fail Criteria**: 
                ### END TEST CASE ###
                """.format(count=missing_neg, error=error_message, solution=solution)
                neg_template = ChatPromptTemplate.from_template(neg_prompt)
                formatted_neg = neg_template.format_prompt().to_messages()
                neg_response = llm.invoke(formatted_neg).content.strip()
                generated_neg = [tc.strip() for tc in re.split(delimiter, neg_response) if tc.strip()]
                generated_neg = generated_neg[:missing_neg]
            
            # Combine CSV test cases with newly generated ones.
            final_pos = (pos_csv + generated_pos)[:required_count]
            final_neg = (neg_csv + generated_neg)[:required_count]

            # Save the newly generated test cases to CSV.
            for tc in generated_pos:
                new_generated_cases.append({
                    "Error": error_message,
                    "Module": module,
                    "TestCase": tc,
                    "Type": "positive"
                })
            for tc in generated_neg:
                new_generated_cases.append({
                    "Error": error_message,
                    "Module": module,
                    "TestCase": tc,
                    "Type": "negative"
                })
            if new_generated_cases:
                save_new_test_cases(new_generated_cases)
        
        # Prepare the output text by labeling each test case.
        test_cases_text = ""
        for idx, tc in enumerate(final_pos, start=1):
            test_cases_text += f"**Positive Test Case {idx}:**\n{tc}\n\n"
        for idx, tc in enumerate(final_neg, start=1):
            test_cases_text += f"**Negative Test Case {idx}:**\n{tc}\n\n"

        response_template = (
            "**Error:**\n{Error}\n\n"
            "**Solution:**\n{Solution}\n\n"
            "**Explanation:**\n{Explanation}\n\n"
            "**Final Test Cases:**\n{TestCases}"
        )
        return {"response": response_template.format(
            Error=error_message,
            Solution=solution,
            Explanation=explanation,
            TestCases=test_cases_text
        )}

    except Exception as e:
        logging.error("Validation/Generation error: %s", str(e))
        return {"response": f"Error processing request: {str(e)}"}

# --------------------------------------------------------------------
# Build the State Graph Workflow
# --------------------------------------------------------------------
workflow = StateGraph(AgentState)
# "retrieve" node: fetch context using the input defect
workflow.add_node("retrieve", lambda state: {"context": retriever.invoke(state["input"])})
workflow.add_node("validate_or_generate_test_cases", validate_or_generate_test_cases)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "validate_or_generate_test_cases")
workflow.add_edge("validate_or_generate_test_cases", END)
agent = workflow.compile()

# --------------------------------------------------------------------
# Automated Evaluation & Self-improvement Functions
# --------------------------------------------------------------------
def auto_evaluate_solution(response: str) -> int:
    if "### END TEST CASE ###" in response:
        return 5
    elif "**Error**:" in response:
        return 1
    else:
        return 3

def generate_alternative_solution(error_message: str) -> str:
    alt_prompt = """
    [INST] Provide a concise, actionable alternative solution for the following error:
    Error: {error}
    Ensure that the solution is clear and does not include any follow-up questions.
    [/INST]
    """
    alt_template = ChatPromptTemplate.from_template(alt_prompt)
    formatted_alt = alt_template.format_prompt(error=error_message)
    alternative_solution = llm.invoke(formatted_alt.to_messages()).content.strip()

    test_case_prompt = """
    [INST] Given the error and the alternative solution:
    Error: {error}
    Solution: {solution}
    Generate EXACTLY 4 structured test cases (2 positive and 2 negative) with the delimiter "### END TEST CASE ###" after each test case.
    Each test case must include:
      - **Test Scenario**
      - **Test Steps**
      - **Pre Requisites**
      - **Expected Results**
      - **Pass/Fail Criteria**
    [/INST]
    """
    tc_template = ChatPromptTemplate.from_template(test_case_prompt)
    formatted_tc = tc_template.format_prompt(error=error_message, solution=alternative_solution)
    alternative_test_cases = llm.invoke(formatted_tc.to_messages()).content.strip()

    alt_response = (
        "**Alternative Solution (Generated):**\n{AltSolution}\n\n"
        "**Test Cases for Alternative Solution:**\n{AltTestCases}"
    ).format(
        AltSolution=alternative_solution,
        AltTestCases=alternative_test_cases
    )
    return alt_response

def get_solution_autonomously(error_message: str) -> str:
    max_iterations = 3
    iteration = 0
    while iteration < max_iterations:
        logging.info("Iteration %d: Processing error: %s", iteration + 1, error_message)
        result = agent.invoke({"input": error_message.strip()})
        response = result["response"]
        logging.info("Agent response:\n%s", response)
        rating = auto_evaluate_solution(response)
        logging.info("Auto-evaluated rating: %d", rating)
        if rating < 3:
            logging.info("Rating below threshold. Generating alternative solution.")
            alt_response = generate_alternative_solution(error_message)
            logging.info("Alternative response generated.")
            return alt_response
        else:
            return response
        iteration += 1
    logging.info("Max iterations reached. Returning last response.")
    return response

# --------------------------------------------------------------------
# Autonomous Agent Execution
# --------------------------------------------------------------------
def main():
    error_description = "Search results not displaying correctly"
    final_solution = get_solution_autonomously(error_description)
    print("\n=== Final Autonomous Response ===\n")
    print(final_solution)

if __name__ == "__main__":
    main()
