from google.colab import userdata

import pandas as pd

from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.documents import Document

from langgraph.graph import END, StateGraph, add_edge

from langchain_core.prompts import ChatPromptTemplate

from langchain_groq import ChatGroq

from typing import TypedDict, List, Optional



# Get API key for Groq

groq_api = userdata.get("groq_api_key")



# File Paths

defect_dump_path = "/mnt/data/Synthetic_Defect_Dump(in).csv"

test_cases_path = "/mnt/data/Amazon_Test_Cases (1)(in) (1).csv"



# Load CSV Files

defect_df = pd.read_csv(defect_dump_path)

test_cases_df = pd.read_csv(test_cases_path)



# Create FAISS vector store for defect descriptions

docs = []

for _, row in defect_df.iterrows():

    if pd.notna(row["Defect Description"]) and pd.notna(row["Steps taken to resolve"]):

        docs.append(Document(

            page_content=row["Defect Description"],

            metadata={

                "solution": row["Steps taken to resolve"],

                "module": row["Module name"]

            }

        ))



# Create Embeddings & FAISS Retriever

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(docs, embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k": 1})



# Define Stateful Agent State

class AgentState(TypedDict):

    input: str

    context: List[dict]

    response: Optional[str]

    step: str

    user_feedback: Optional[str]



# Initialize Groq LLM

llm = ChatGroq(

    groq_api_key=groq_api,

    temperature=0.3,

    model_name="llama-3.2-1b-preview",

)



# Step 1: Retrieve Defect Information

def retrieve(state: AgentState):

    relevant_docs = retriever.invoke(state["input"])

    return {"context": relevant_docs, "step": "provide_solution"}



# Step 2: Fetch Test Cases & Provide Solution

def provide_solution(state: AgentState):

    if not state["context"] or "solution" not in state["context"][0].metadata:

        return {"response": "**Error:** The error message is unknown, and test cases cannot be generated.", "step": "end"}



    context = state["context"][0]

    module_name = context.metadata["module"]

    solution_text = context.metadata["solution"]



    # Fetch relevant test cases (4 test cases)

    test_cases = test_cases_df[test_cases_df["Module name"] == module_name].head(4)

    test_cases_list = []

    for _, tc in test_cases.iterrows():

        test_cases_list.append(f"1. **Test Case ID:** {tc['Test Case ID']}\n"

                               f"   - **Description:** {tc['Test Description']}\n"

                               f"   - **Steps:** {tc['Test Steps']}\n")



    test_cases_text = "\n".join(test_cases_list) if test_cases_list else "No test cases found."



    return {

        "response": f"**Error:** {state['input']}\n\n**Solution:** {solution_text}\n\n**Test Cases:**\n{test_cases_text}",

        "step": "ask_feedback"

    }



# Step 3: Ask for User Feedback

def ask_feedback(state: AgentState):

    return {

        "response": "Have you tried the suggested solution? Reply with **yes** if it worked, **no** if you need more help.",

        "step": "handle_feedback"

    }



# Step 4: Handle User Feedback

def handle_feedback(state: AgentState):

    user_response = state.get("user_feedback", "").strip().lower()

    if user_response == "yes":

        return {"response": "Glad the issue is resolved! Let me know if you need anything else.", "step": "end"}

    elif user_response == "no":

        return {

            "response": "Can you describe what issue you're still facing? I'll suggest alternative fixes.",

            "step": "ask_alternative_fix"

        }

    else:

        return {"response": "Please reply with **yes** or **no**.", "step": "handle_feedback"}



# Step 5: Suggest Alternative Fixes (Using LLM)

def suggest_alternative_fix(state: AgentState):

    context = state["context"][0] if state["context"] else None

    solution_text = context.metadata["solution"] if context else "No known solution available."

    

    prompt_template = """[INST] Given the initial solution:

    {solution}



    The user is still facing issues. Suggest alternative debugging steps or fixes. [/INST]"""



    alternative_fix = llm.invoke(ChatPromptTemplate.from_template(prompt_template).format(

        solution=solution_text

    )).content



    return {"response": f"Here are alternative suggestions:\n\n{alternative_fix}", "step": "ask_feedback"}



# Define LangGraph Workflow

workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve)

workflow.add_node("provide_solution", provide_solution)

workflow.add_node("ask_feedback", ask_feedback)

workflow.add_node("handle_feedback", handle_feedback)

workflow.add_node("ask_alternative_fix", suggest_alternative_fix)



# Define transitions

workflow.set_entry_point("retrieve")

add_edge(workflow, "retrieve", "provide_solution")

add_edge(workflow, "provide_solution", "ask_feedback")

add_edge(workflow, "ask_feedback", "handle_feedback")

add_edge(workflow, "handle_feedback", "ask_alternative_fix", condition=lambda state: state["user_feedback"] == "no")

add_edge(workflow, "handle_feedback", END, condition=lambda state: state["user_feedback"] == "yes")

add_edge(workflow, "ask_alternative_fix", "ask_feedback")



# Compile the chatbot agent

agent = workflow.compile()



# Function to interact with the chatbot

def chat_with_agent(error_message):

    state = {"input": error_message.strip(), "step": "retrieve", "user_feedback": None}

    while state["step"] != "end":

        result = agent.invoke(state)

        print(result["response"])

        if result["step"] in ["handle_feedback", "ask_alternative_fix"]:

            user_input = input("Your response: ")

            result["user_feedback"] = user_input

        state = result



# Example Interaction

print("\n=== AI Chatbot for Error Resolution ===")

chat_with_agent("Application crash on logout")

```



---



### **How This is Now an Agentic AI Chatbot:**

✅ **Multi-step Interaction:** It doesn’t stop after returning an answer but guides the user.  

✅ **Memory & Context Awareness:** It tracks user feedback and adjusts responses accordingly.  

✅ **Decision-Making:** If the solution doesn’t work, it suggests alternative fixes automatically.  

✅ **Stateful Conversation:** The chatbot "remembers" the step it's in and moves forward accordingly.  



### **How to Use:**

1. Run the script in **Google Colab**.  

2. Call `chat_with_agent("Your error message")`.  

3. The AI will **retrieve the solution & test cases, then interactively help you resolve the issue**.  



Let me know if you need further improvements! 🚀
