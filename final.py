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


# defining a similarity function
from difflib import SequenceMatcher

def similarity(a: str, b:str) -> float:
  return SequenceMatcher(None, a, b).ratio()

# ------------------------------------------------------------------------------
# Setup Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------------------------------------------------------
# Data Loading and Document Preparation
# ------------------------------------------------------------------------------
groq_api = userdata.get("groq_api_key")
# Load defects file (expected columns: Module, Description, Solution, Severity_Level, Affected_Area, Detection_Date, Resolved_By)
df = pd.read_csv("/content/defects.csv")

# Load the test cases CSV.
# Expected columns: Module, Test_Scenario, Test_Steps, Pre_Requisite, Pass_Fail_Criteria, Expected_Result
try:
    test_cases_df = pd.read_csv("/content/test_cases.csv")
except Exception as e:
    logging.warning("Test cases file not found or unreadable. Creating an empty DataFrame.")
    test_cases_df = pd.DataFrame(columns=["Module", "Test_Scenario", "Test_Steps", "Pre_Requisite", "Pass_Fail_Criteria", "Expected_Result"])

# Build documents from defects CSV (only using Module, Description, Solution)
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

# Create embeddings and FAISS vector store for retrieval
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

# ------------------------------------------------------------------------------
# Helper: Classify Test Case (Positive/Negative) based on keywords
# ------------------------------------------------------------------------------
def classify_test_case(tc_text: str) -> str:
    negative_keywords = [
        "fail", "error", "misconfigured", "incorrect", "doesn't work",
        "not work", "negative", "invalid", "wrong", "missing", "unexpected",
        "should not", "incorrectly", "failure", "reject", "malformed",
        "timeout", "invalid input", "edge case", "out of bounds"
    ]
    text_lower = tc_text.lower()
    return "negative" if any(kw in text_lower for kw in negative_keywords) else "positive"

# ------------------------------------------------------------------------------
# Helper: Retrieve CSV Test Cases by Module
# ------------------------------------------------------------------------------
def get_csv_test_cases(module: str):
    """
    Returns a tuple of two lists: (positive_test_cases, negative_test_cases).
    Each test case is constructed from the CSV columns and validated.
    """
    positive_cases = []
    negative_cases = []
    
    global test_cases_df
    if test_cases_df.empty:
        return positive_cases, negative_cases

    # Filter by Module
    df_filtered = test_cases_df[test_cases_df["Module"] == module]
    
    for _, row in df_filtered.iterrows():
        # Combine the five columns into one text block for classification
        tc_text = " ".join([
            str(row.get("Test_Scenario", "")),
            str(row.get("Test_Steps", "")),
            str(row.get("Pre_Requisite", "")),
            str(row.get("Expected_Result", "")),
            str(row.get("Pass_Fail_Criteria", ""))
        ])
        if tc_text.strip() == "":
            continue
        tc_type = classify_test_case(tc_text)
        # Also store the row as a dictionary for later formatting
        tc_dict = {
            "Module": row["Module"],
            "Test_Scenario": row["Test_Scenario"],
            "Test_Steps": row["Test_Steps"],
            "Pre_Requisite": row["Pre_Requisite"],
            "Pass_Fail_Criteria": row["Pass_Fail_Criteria"],
            "Expected_Result": row["Expected_Result"]
        }
        if tc_type == "positive":
            positive_cases.append(tc_dict)
        else:
            negative_cases.append(tc_dict)
    return positive_cases, negative_cases

# ------------------------------------------------------------------------------
# Helper: Parse Generated Test Case into Fields
# ------------------------------------------------------------------------------
def parse_test_case(tc_text: str) -> dict:
    """
    Parses a generated test case text to extract the fields.
    Expected labels: Test_Scenario:, Test_Steps:, Pre_Requisite:, Expected_Result:, Pass_Fail_Criteria:
    """
    fields = {
        "Test_Scenario": "",
        "Test_Steps": "",
        "Pre_Requisite": "",
        "Expected_Result": "",
        "Pass_Fail_Criteria": ""
    }
    # Use regex to capture content after each label up to the next label or end.
    for field in fields.keys():
        # Pattern looks for e.g. "Test_Scenario:" then capture until the next field label
        pattern = field + r":\s*(.*?)\s*(?=(Test_Steps:|Pre_Requisite:|Expected_Result:|Pass_Fail_Criteria:|$))"
        match = re.search(pattern, tc_text, re.DOTALL)
        if match:
            fields[field] = match.group(1).strip()
    return fields

# ------------------------------------------------------------------------------
# Helper: Save New Test Cases to CSV (avoiding duplicates)
# ------------------------------------------------------------------------------
def save_new_test_cases(new_cases: List[dict]):
    """
    new_cases: list of dictionaries with keys:
      Module, Test_Scenario, Test_Steps, Pre_Requisite, Pass_Fail_Criteria, Expected_Result.
    Appends only new (non-duplicate) test cases to the CSV.
    """
    global test_cases_df
    required_columns = ["Module", "Test_Scenario", "Test_Steps", "Pre_Requisite", "Pass_Fail_Criteria", "Expected_Result"]
    if test_cases_df.empty:
        test_cases_df = pd.DataFrame(columns=required_columns)
    
    rows_to_add = []
    for case in new_cases:
        # Consider a test case duplicate if all fields match for the same module.
        duplicate = test_cases_df[
            (test_cases_df["Module"] == case["Module"]) &
            (test_cases_df["Test_Scenario"] == case["Test_Scenario"]) &
            (test_cases_df["Test_Steps"] == case["Test_Steps"]) &
            (test_cases_df["Pre_Requisite"] == case["Pre_Requisite"]) &
            (test_cases_df["Pass_Fail_Criteria"] == case["Pass_Fail_Criteria"]) &
            (test_cases_df["Expected_Result"] == case["Expected_Result"])
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

# ------------------------------------------------------------------------------
# Define Agent State and LLM Initialization
# ------------------------------------------------------------------------------
class AgentState(TypedDict):
    input: str
    context: List[Document]
    response: str

llm = ChatGroq(
    groq_api_key=groq_api,
    temperature=0.3,
    model_name="gemma2-9b-it",
)

# ------------------------------------------------------------------------------
# Workflow Node: Validate or Generate Test Cases (with CSV storage)
# ------------------------------------------------------------------------------
def validate_or_generate_test_cases(state: AgentState):
    try:
        if not state["context"]:
            return {"response": "**Error**: The defect could not be found in the database."}
        context = state["context"][0]
        error_message = state["input"]
        # new changes start
        if similarity(error_message.lower(), context.page_content.lower())<0.3:
          return {"response": "**Error**:The defect could not be found in database."}
          # new changes end
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

        required_count = 2  # Require 2 positive and 2 negative test cases

        # Fetch test cases from CSV (filtered by module)
        pos_csv, neg_csv = get_csv_test_cases(module)
        logging.info("Fetched %d positive and %d negative CSV test cases.", len(pos_csv), len(neg_csv))

        new_generated_cases = []  # To collect any new generated cases for saving

        # If sufficient CSV test cases exist, use them directly.
        if len(pos_csv) >= required_count and len(neg_csv) >= required_count:
            final_pos = pos_csv[:required_count]
            final_neg = neg_csv[:required_count]
        else:
            # Determine how many test cases are missing
            missing_pos = max(0, required_count - len(pos_csv))
            missing_neg = max(0, required_count - len(neg_csv))
            generated_pos = []
            generated_neg = []
            delimiter = "\n### END TEST CASE ###\n"

            # Prompt for missing positive test cases.
            if missing_pos > 0:
                pos_prompt = """
                [INST] Generate EXACTLY {count} POSITIVE test case(s) for:
                Error: {error}
                Solution: {solution}

                Each test case MUST include the following sections and end with the delimiter "### END TEST CASE ###":
                Test_Scenario: A short description of the scenario.
                Test_Steps: Step-by-step instructions.
                Pre_Requisite: Conditions before running the test.
                Expected_Result: What should happen if the solution works.
                Pass_Fail_Criteria: How to determine if the test passes.

                Output format (including the delimiter):
                1. Test_Scenario: 
                   Test_Steps: 
                   Pre_Requisite: 
                   Expected_Result: 
                   Pass_Fail_Criteria: 
                ### END TEST CASE ###
                """.format(count=missing_pos, error=error_message, solution=solution)
                pos_template = ChatPromptTemplate.from_template(pos_prompt)
                formatted_pos = pos_template.format_prompt().to_messages()
                pos_response = llm.invoke(formatted_pos).content.strip()
                # Split using the delimiter and parse each generated test case.
                pos_cases_raw = [tc.strip() for tc in re.split(delimiter, pos_response) if tc.strip()]
                for tc_raw in pos_cases_raw[:missing_pos]:
                    parsed = parse_test_case(tc_raw)
                    if parsed["Test_Scenario"]:  # basic check that parsing worked
                        generated_pos.append({
                            "Module": module,
                            "Test_Scenario": parsed["Test_Scenario"],
                            "Test_Steps": parsed["Test_Steps"],
                            "Pre_Requisite": parsed["Pre_Requisite"],
                            "Pass_Fail_Criteria": parsed["Pass_Fail_Criteria"],
                            "Expected_Result": parsed["Expected_Result"]
                        })
            
            # Prompt for missing negative test cases.
            if missing_neg > 0:
                neg_prompt = """
                [INST] Generate EXACTLY {count} NEGATIVE test case(s) for:
                Error: {error}
                Solution: {solution}

                Each test case MUST include the following sections and end with the delimiter "### END TEST CASE ###":
                Test_Scenario: A short description of the scenario.
                Test_Steps: Step-by-step instructions.
                Pre_Requisite: Conditions before running the test.
                Expected_Result: What should happen if the solution fails.
                Pass_Fail_Criteria: How to determine if the test fails.

                Output format (including the delimiter):
                1. Test_Scenario: 
                   Test_Steps: 
                   Pre_Requisite: 
                   Expected_Result: 
                   Pass_Fail_Criteria: 
                ### END TEST CASE ###
                """.format(count=missing_neg, error=error_message, solution=solution)
                neg_template = ChatPromptTemplate.from_template(neg_prompt)
                formatted_neg = neg_template.format_prompt().to_messages()
                neg_response = llm.invoke(formatted_neg).content.strip()
                neg_cases_raw = [tc.strip() for tc in re.split(delimiter, neg_response) if tc.strip()]
                for tc_raw in neg_cases_raw[:missing_neg]:
                    parsed = parse_test_case(tc_raw)
                    if parsed["Test_Scenario"]:
                        generated_neg.append({
                            "Module": module,
                            "Test_Scenario": parsed["Test_Scenario"],
                            "Test_Steps": parsed["Test_Steps"],
                            "Pre_Requisite": parsed["Pre_Requisite"],
                            "Pass_Fail_Criteria": parsed["Pass_Fail_Criteria"],
                            "Expected_Result": parsed["Expected_Result"]
                        })

            # Combine CSV test cases with newly generated ones (if CSV ones are available)
            final_pos = (pos_csv + generated_pos)[:required_count]
            final_neg = (neg_csv + generated_neg)[:required_count]

            # Save any new generated test cases to the CSV.
            new_generated_cases.extend(generated_pos)
            new_generated_cases.extend(generated_neg)
            if new_generated_cases:
                save_new_test_cases(new_generated_cases)
        
        # Build the final test cases output text.
        def format_tc(tc: dict) -> str:
            return (f"Test_Scenario: {tc['Test_Scenario']}\n"
                    f"Test_Steps: {tc['Test_Steps']}\n"
                    f"Pre_Requisite: {tc['Pre_Requisite']}\n"
                    f"Expected_Result: {tc['Expected_Result']}\n"
                    f"Pass_Fail_Criteria: {tc['Pass_Fail_Criteria']}")
        
        test_cases_text = ""
        for idx, tc in enumerate(final_pos, start=1):
            test_cases_text += f"**Positive Test Case {idx}:**\n{format_tc(tc)}\n\n"
        for idx, tc in enumerate(final_neg, start=1):
            test_cases_text += f"**Negative Test Case {idx}:**\n{format_tc(tc)}\n\n"

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

# ------------------------------------------------------------------------------
# Build the State Graph Workflow
# ------------------------------------------------------------------------------
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", lambda state: {"context": retriever.invoke(state["input"])})
workflow.add_node("validate_or_generate_test_cases", validate_or_generate_test_cases)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "validate_or_generate_test_cases")
workflow.add_edge("validate_or_generate_test_cases", END)
agent = workflow.compile()

# ------------------------------------------------------------------------------
# Automated Evaluation & Self-improvement Functions
# ------------------------------------------------------------------------------
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
      Test_Scenario
      Test_Steps
      Pre_Requisite
      Expected_Result
      Pass_Fail_Criteria
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

# ------------------------------------------------------------------------------
# Autonomous Agent Execution
# ------------------------------------------------------------------------------
def main():
    error_description = "Search results not displaying correctly"
    final_solution = get_solution_autonomously(error_description)
    print("\n=== Final Autonomous Response ===\n")
    print(final_solution)

if __name__ == "__main__":
    main()
