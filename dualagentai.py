!pip install langgraph langchain pandas faiss-cpu langchain-groq sentence-transformers langchain_community

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from langchain_groq import ChatGroq
from typing import TypedDict, List

# -----------------------------------------------------
# Data Loading and Preparation
# -----------------------------------------------------

# Load system defects and test cases
df_defects = pd.read_csv("/content/system_defects.csv")
df_test_cases = pd.read_csv("/content/test_cases.csv")

# Prepare documents for retrieval.
# Embed the module name into the page content so that the vector representation is more specific.
docs = []
for _, row in df_defects.iterrows():
    if pd.notna(row["Defect Description"]) and pd.notna(row["Steps taken to resolve"]):
        combined_text = f"Module: {row['Module name']}\nError: {row['Defect Description']}"
        docs.append(Document(
            page_content=combined_text,
            metadata={"solution": row["Steps taken to resolve"], "module": row["Module name"]}
        ))

# Create vector store for retrieval using a sentence-transformer model.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

# -----------------------------------------------------
# Defect Resolution Workflow
# -----------------------------------------------------

# Define the AgentState for the defect workflow.
class AgentState(TypedDict):
    input: str
    context: List[dict]
    response: str

# Setup the Groq LLM (replace with your actual API key)
groq_api = "your_groq_api_key"
llm = ChatGroq(
    groq_api_key=groq_api,
    temperature=0.3,
    model_name="gemma2-9b-it",
)

def retrieve(state: AgentState):
    # The retrieval now uses the state["input"] (which may include module information).
    relevant_docs = retriever.invoke(state["input"])
    return {"context": relevant_docs} if relevant_docs else {"context": []}

def fetch_test_cases(module_name: str):
    module_cases = df_test_cases[df_test_cases["Module name"] == module_name]
    return module_cases.sample(n=min(4, len(module_cases))).to_dict(orient="records") if not module_cases.empty else []

def generate_response(state: AgentState):
    if state["context"] and "solution" in state["context"][0].metadata:
        context_doc = state["context"][0]
        test_cases = fetch_test_cases(context_doc.metadata["module"])
        
        response_template = """**Error:**\n{Error}\n\n**Solution:**\n{Solution}\n\n**Test Cases:**\n{TestCases}"""
        formatted_cases = "\n\n".join([
            f"**Test Case ID:** {tc['Test Case ID']}\n**Scenario:** {tc['Test Description']}\n**Steps:** {tc['Test Steps']}\n**Expected Result:** {tc['Expected Results']}"
            for tc in test_cases
        ])
        
        return {"response": response_template.format(
            Error=state["input"],
            Solution=context_doc.metadata["solution"],
            TestCases=formatted_cases if formatted_cases else "No relevant test cases found."
        )}
    return {"response": "**Error:** The defect is unknown and cannot be resolved."}

workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate_response", generate_response)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate_response")
workflow.add_edge("generate_response", END)
agent_defect = workflow.compile()

def get_solution(defect_description: str) -> str:
    """
    Processes the defect description.
    If the defect input is in the form "<module>: <defect text>", then the module is extracted
    and prepended to the query so that retrieval is module-specific.
    """
    module_filter = None
    # Split on the first colon to see if a module is provided.
    parts = defect_description.split(":", 1)
    known_modules = {m.lower() for m in df_defects["Module name"].dropna().unique()}
    if len(parts) == 2:
        potential_module = parts[0].strip().lower()
        if potential_module in known_modules:
            module_filter = potential_module
            defect_description = parts[1].strip()
    
    # Build the query text. If a module filter is provided, include it.
    if module_filter:
        query_text = f"Module: {module_filter}\nError: {defect_description}"
    else:
        query_text = defect_description

    state = AgentState(input=query_text, context=[], response="")
    result = agent_defect.invoke(state)
    return result["response"]

# -----------------------------------------------------
# LLM-Based Q&A for Additional Queries
# -----------------------------------------------------

def get_llm_response(query: str) -> str:
    prompt = f"Answer the following question as helpfully as possible:\n\nQuestion: {query}"
    result = llm.invoke(prompt)
    return result.content

# -----------------------------------------------------
# Chat Interface with Memory & Feedback
# -----------------------------------------------------

class ChatAgent:
    def __init__(self):
        self.conversation_history = []  # List to store conversation turns.
        self.current_defect_context = None  # Stores the most recent defect query (including module info if provided).
    
    def process_input(self, user_input: str) -> str:
        user_input = user_input.strip()
        
        # Check for feedback.
        if user_input.lower().startswith("feedback:"):
            feedback = user_input[len("feedback:"):].strip()
            return self.provide_feedback(feedback)
        
        # Check for follow-up command to elaborate on the current defect.
        elif user_input.lower() in ["elaborate", "continue", "more"]:
            if self.current_defect_context:
                # Reuse the stored defect context.
                response = get_solution(self.current_defect_context)
                self.conversation_history.append({"role": "agent", "message": response})
                return response
            else:
                return "No previous defect context to elaborate on."
        
        # Determine if this is a defect description (prefixed with "defect:") or a general question.
        elif user_input.lower().startswith("defect:"):
            defect_query = user_input[len("defect:"):].strip()
            self.current_defect_context = defect_query  # Store for follow-up.
            response = get_solution(defect_query)
        elif user_input.lower().startswith("question:"):
            query = user_input[len("question:"):].strip()
            response = get_llm_response(query)
        else:
            # Default to treating the input as a general question.
            response = get_llm_response(user_input)
        
        self.conversation_history.append({"role": "user", "message": user_input})
        self.conversation_history.append({"role": "agent", "message": response})
        return response

    def provide_feedback(self, feedback: str) -> str:
        self.conversation_history.append({"role": "feedback", "message": feedback})
        ack = "Thank you for your feedback. I'll take it into account."
        self.conversation_history.append({"role": "agent", "message": ack})
        return ack

    def show_history(self):
        for turn in self.conversation_history:
            print(f"{turn['role'].capitalize()}: {turn['message']}\n")

# -----------------------------------------------------
# Chat Loop Example
# -----------------------------------------------------

if __name__ == "__main__":
    chat_agent = ChatAgent()
    print("Welcome to the Agentic Defect Resolution & Q&A Chatbot!")
    print("Usage:")
    print("  - To get a defect solution, type: defect: <module>: <your defect description>")
    print("    (e.g., 'defect: cart: Application crash on logout')")
    print("  - To ask a general question, type: question: <your query>")
    print("    or simply type your query without a prefix.")
    print("  - To provide feedback, type: feedback: <your feedback>")
    print("  - To elaborate on the last defect (without retyping), type: elaborate")
    print("  - To view conversation history, type: history")
    print("  - Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        elif user_input.lower() == "history":
            chat_agent.show_history()
        else:
            response = chat_agent.process_input(user_input)
            print(f"Agent: {response}\n")















# -*- coding: utf-8 -*-
"""DualAgentAI.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1pKyZE2GECKj4aEIIagluNQduNZjimBoU
"""

!pip install langgraph langchain pandas faiss-cpu langchain-groq sentence-transformers langchain_community

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from typing import TypedDict, List

# ---------------------------
# Load Data and Prepare Docs
# ---------------------------

# Load system defects and test cases
df_defects = pd.read_csv("/content/system_defects.csv")
df_test_cases = pd.read_csv("/content/test_cases.csv")

# Prepare documents for retrieval
docs = []
for _, row in df_defects.iterrows():
    if pd.notna(row["Defect Description"]) and pd.notna(row["Steps taken to resolve"]):
        docs.append(Document(
            page_content=row["Defect Description"],
            metadata={"solution": row["Steps taken to resolve"], "module": row["Module name"]}
        ))

# Create vector store for retrieval
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

# ---------------------------
# Define Workflow Nodes
# ---------------------------

# Extend the agent state to include conversation history.
class AgentState(TypedDict):
    input: str
    context: List[dict]
    response: str
    conversation_history: List[dict]

# Setup your Groq LLM (replace with your actual API key)
from google.colab import userdata
groq_api = userdata.get("groq_api_key")
llm = ChatGroq(
    groq_api_key=groq_api,
    temperature=0.3,
    model_name="gemma2-9b-it",
)

# This function retrieves relevant documents based on the user input.
def retrieve(state: AgentState):
    relevant_docs = retriever.invoke(state["input"])
    return {"context": relevant_docs} if relevant_docs else {"context": []}

# Function to fetch some test cases for the given module
def fetch_test_cases(module_name: str):
    module_cases = df_test_cases[df_test_cases["Module name"] == module_name]
    return module_cases.sample(n=min(4, len(module_cases))).to_dict(orient="records") if not module_cases.empty else []

# This function generates the answer including the defect solution and sample test cases.
def generate_response(state: AgentState):
    if state["context"] and "solution" in state["context"][0].metadata:
        context_doc = state["context"][0]
        test_cases = fetch_test_cases(context_doc.metadata["module"])

        response_template = """**Error:**\n{Error}\n\n**Solution:**\n{Solution}\n\n**Test Cases:**\n{TestCases}"""

        formatted_cases = "\n\n".join([
            f"**Test Case ID:** {tc['Test Case ID']}\n**Scenario:** {tc['Test Description']}\n**Steps:** {tc['Test Steps']}\n**Expected Result:** {tc['Expected Results']}"
            for tc in test_cases
        ])

        return {"response": response_template.format(
            Error=state["input"],
            Solution=context_doc.metadata["solution"],
            TestCases=formatted_cases if formatted_cases else "No relevant test cases found."
        )}
    # Fallback if no context found
    return {"response": "**Error:** The defect is unknown and cannot be resolved."}

# Build the agent workflow graph.
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate_response", generate_response)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate_response")
workflow.add_edge("generate_response", END)
agent = workflow.compile()

# ---------------------------
# Define ChatAgent Class
# ---------------------------

class ChatAgent:
    def __init__(self):
        # Maintain a list of conversation turns; each turn is a dict with role and message.
        self.conversation_history = []

    def process_input(self, user_input: str):
        # Update history with user message
        self.conversation_history.append({"role": "user", "message": user_input})

        # Build an agent state that includes conversation history.
        # (For now, we pass only the current defect input to the agent.
        #  You can later enhance retrieval or response by including previous context.)
        state = AgentState(
            input=user_input,
            context=[],  # Will be filled in the workflow
            response="",
            conversation_history=self.conversation_history.copy()
        )

        # Invoke the workflow
        result = agent.invoke(state)
        response = result["response"]

        # Append agent's response to the conversation history
        self.conversation_history.append({"role": "agent", "message": response})
        return response

    def provide_feedback(self, feedback: str):
        # Append feedback into the conversation history.
        self.conversation_history.append({"role": "feedback", "message": feedback})
        # You might use this feedback to adjust the context or call a follow-up chain.
        # For now, we just acknowledge the feedback.
        ack = "Thanks for your feedback! I will try to improve based on your input."
        self.conversation_history.append({"role": "agent", "message": ack})
        return ack

    def show_history(self):
        # Utility function to see the conversation log.
        for turn in self.conversation_history:
            role = turn["role"]
            msg = turn["message"]
            print(f"{role.capitalize()}: {msg}\n")

# ---------------------------
# Chat Loop Example
# ---------------------------

if __name__ == "__main__":
    chat_agent = ChatAgent()
    print("Welcome to the Defect Resolution Chatbot! (Type 'exit' to quit)")

    while True:
        # Read user input. If the input starts with "feedback:" treat it as feedback.
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        elif user_input.lower().startswith("feedback:"):
            # Extract the feedback content and process it.
            feedback = user_input[len("feedback:"):].strip()
            ack = chat_agent.provide_feedback(feedback)
            print(f"Agent: {ack}")
        else:
            # Process the defect or question normally.
            response = chat_agent.process_input(user_input)
            print(f"Agent: {response}")

!pip install langgraph langchain pandas faiss-cpu langchain-groq sentence-transformers langchain_community

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from langchain_groq import ChatGroq
from typing import TypedDict, List

# -----------------------------------------------------
# Data Loading and Preparation (Same as Original Code)
# -----------------------------------------------------

# Load system defects and test cases
df_defects = pd.read_csv("/content/system_defects.csv")
df_test_cases = pd.read_csv("/content/test_cases.csv")

# Prepare documents for retrieval: each defect document holds a solution and module.
docs = []
for _, row in df_defects.iterrows():
    if pd.notna(row["Defect Description"]) and pd.notna(row["Steps taken to resolve"]):
        docs.append(Document(
            page_content=row["Defect Description"],
            metadata={"solution": row["Steps taken to resolve"], "module": row["Module name"]}
        ))

# Create vector store for retrieval using sentence-transformers
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

# -----------------------------------------------------
# Original Agent: Defect -> Solution & Test Cases Workflow
# -----------------------------------------------------

# Define the AgentState for the defect workflow.
class AgentState(TypedDict):
    input: str
    context: List[dict]
    response: str

# Setup the Groq LLM for both our workflows (replace with your actual API key)
from google.colab import userdata
groq_api = userdata.get("groq_api_key")
llm = ChatGroq(
    groq_api_key=groq_api,
    temperature=0.3,
    model_name="gemma2-9b-it",
)

def retrieve(state: AgentState):
    relevant_docs = retriever.invoke(state["input"])
    return {"context": relevant_docs} if relevant_docs else {"context": []}

def fetch_test_cases(module_name: str):
    module_cases = df_test_cases[df_test_cases["Module name"] == module_name]
    return module_cases.sample(n=min(4, len(module_cases))).to_dict(orient="records") if not module_cases.empty else []

def generate_response(state: AgentState):
    if state["context"] and "solution" in state["context"][0].metadata:
        context_doc = state["context"][0]
        test_cases = fetch_test_cases(context_doc.metadata["module"])

        response_template = """**Error:**\n{Error}\n\n**Solution:**\n{Solution}\n\n**Test Cases:**\n{TestCases}"""

        formatted_cases = "\n\n".join([
            f"**Test Case ID:** {tc['Test Case ID']}\n**Scenario:** {tc['Test Description']}\n**Steps:** {tc['Test Steps']}\n**Expected Result:** {tc['Expected Results']}"
            for tc in test_cases
        ])

        return {"response": response_template.format(
            Error=state["input"],
            Solution=context_doc.metadata["solution"],
            TestCases=formatted_cases if formatted_cases else "No relevant test cases found."
        )}
    return {"response": "**Error:** The defect is unknown and cannot be resolved."}

# Create the workflow graph for the defect resolution agent.
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate_response", generate_response)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate_response")
workflow.add_edge("generate_response", END)
agent_defect = workflow.compile()

def get_solution(defect_description: str) -> str:
    """
    Given a defect description, this function returns the solution and test cases.
    """
    state = AgentState(input=defect_description.strip(), context=[], response="")
    result = agent_defect.invoke(state)
    return result["response"]

# # -----------------------------------------------------
# # New Agent: LLM-Based Q&A for Additional Issues or Questions
# # -----------------------------------------------------

# def get_llm_response(query: str) -> str:
#     """
#     Uses the ChatGroq LLM to generate a response to a general query.
#     """
#     # Here you can build a more elaborate prompt using conversation history if needed.
#     prompt = f"Answer the following question as helpfully as possible:\n\nQuestion: {query}"
#     # The llm.invoke call returns a dict with the key "response"
#     result = llm.invoke({"input": prompt})
#     return result["response"]

# # -----------------------------------------------------
# # Chat Interface: Routing between Defect Agent and LLM Q&A Agent
# # -----------------------------------------------------

# class ChatAgent:
#     def __init__(self):
#         # Store the conversation history if needed.
#         self.conversation_history = []

#     def process_input(self, user_input: str) -> str:
#         """
#         Processes the user input by routing to the defect resolution agent or the LLM Q&A agent.
#         Use the prefix "defect:" for defect descriptions and "question:" (or no prefix) for general questions.
#         """
#         user_input = user_input.strip()
#         # Check for prefix to determine which agent to call.
#         if user_input.lower().startswith("defect:"):
#             # Remove the prefix and get solution from the defect agent.
#             defect_query = user_input[len("defect:"):].strip()
#             response = get_solution(defect_query)
#         elif user_input.lower().startswith("question:"):
#             # Remove the prefix and get answer from the LLM-based Q&A agent.
#             query = user_input[len("question:"):].strip()
#             response = get_llm_response(query)
#         else:
#             # If no prefix is provided, you can choose a default behavior.
#             # For example, assume it is a general query.
#             response = get_llm_response(user_input)

#         # Optionally, record the conversation history.
#         self.conversation_history.append({"role": "user", "message": user_input})
#         self.conversation_history.append({"role": "agent", "message": response})
#         return response

#     def show_history(self):
#         """
#         Utility to print the conversation history.
#         """
#         for turn in self.conversation_history:
#             print(f"{turn['role'].capitalize()}: {turn['message']}\n")

# # -----------------------------------------------------
# # Chat Loop Example
# # -----------------------------------------------------

# if __name__ == "__main__":
#     chat_agent = ChatAgent()
#     print("Welcome to the Hybrid Defect Resolution & Q&A Chatbot!")
#     print("Type 'defect: <your defect description>' to get a solution with test cases.")
#     print("Type 'question: <your query>' for general questions (e.g., issues with the solution, clarifications, etc.).")
#     print("Type 'exit' to quit.\n")

#     while True:
#         user_input = input("You: ").strip()
#         if user_input.lower() == "exit":
#             print("Goodbye!")
#             break
#         response = chat_agent.process_input(user_input)
#         print(f"Agent: {response}\n")

# -----------------------------------------------------
# New Agent: LLM-Based Q&A for Additional Issues or Questions
# -----------------------------------------------------

def get_llm_response(query: str) -> str:
    """
    Uses the ChatGroq LLM to generate a response to a general query.
    Note: Pass a string prompt directly.
    """
    prompt = f"Answer the following question as helpfully as possible:\n\nQuestion: {query}"
    # Pass the prompt string directly rather than a dict.
    result = llm.invoke(prompt)
    return result["response"]

# -----------------------------------------------------
# Chat Interface: Routing between Defect Agent and LLM Q&A Agent
# -----------------------------------------------------

class ChatAgent:
    def __init__(self):
        # Store the conversation history if needed.
        self.conversation_history = []

    def process_input(self, user_input: str) -> str:
        """
        Processes the user input by routing to the defect resolution agent or the LLM Q&A agent.
        Use the prefix "defect:" for defect descriptions and "question:" (or no prefix) for general questions.
        """
        user_input = user_input.strip()
        # Check for prefix to determine which agent to call.
        if user_input.lower().startswith("defect:"):
            # Remove the prefix and get solution from the defect agent.
            defect_query = user_input[len("defect:"):].strip()
            response = get_solution(defect_query)
        elif user_input.lower().startswith("question:"):
            # Remove the prefix and get answer from the LLM-based Q&A agent.
            query = user_input[len("question:"):].strip()
            response = get_llm_response(query)
        else:
            # If no prefix is provided, assume it's a general query.
            response = get_llm_response(user_input)

        # Optionally, record the conversation history.
        self.conversation_history.append({"role": "user", "message": user_input})
        self.conversation_history.append({"role": "agent", "message": response})
        return response

    def show_history(self):
        """
        Utility to print the conversation history.
        """
        for turn in self.conversation_history:
            print(f"{turn['role'].capitalize()}: {turn['message']}\n")

# -----------------------------------------------------
# Chat Loop Example
# -----------------------------------------------------

if __name__ == "__main__":
    chat_agent = ChatAgent()
    print("Welcome to the Hybrid Defect Resolution & Q&A Chatbot!")
    print("Type 'defect: <your defect description>' to get a solution with test cases.")
    print("Type 'question: <your query>' for general questions (e.g., issues with the solution, clarifications, etc.).")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = chat_agent.process_input(user_input)
        print(f"Agent: {response}\n")

!pip install langgraph langchain pandas faiss-cpu langchain-groq sentence-transformers langchain_community

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from langchain_groq import ChatGroq
from typing import TypedDict, List

# -----------------------------------------------------
# Data Loading and Preparation (Same as Original Code)
# -----------------------------------------------------

# Load system defects and test cases
df_defects = pd.read_csv("/content/system_defects.csv")
df_test_cases = pd.read_csv("/content/test_cases.csv")

# Prepare documents for retrieval: each defect document holds a solution and module.
docs = []
for _, row in df_defects.iterrows():
    if pd.notna(row["Defect Description"]) and pd.notna(row["Steps taken to resolve"]):
        docs.append(Document(
            page_content=row["Defect Description"],
            metadata={"solution": row["Steps taken to resolve"], "module": row["Module name"]}
        ))

# Create vector store for retrieval using sentence-transformers
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

# -----------------------------------------------------
# Original Agent: Defect -> Solution & Test Cases Workflow
# -----------------------------------------------------

# Define the AgentState for the defect workflow.
class AgentState(TypedDict):
    input: str
    context: List[dict]
    response: str

# Setup the Groq LLM for both our workflows (replace with your actual API key)
from google.colab import userdata
groq_api = userdata.get("groq_api_key")
llm = ChatGroq(
    groq_api_key=groq_api,
    temperature=0.3,
    model_name="gemma2-9b-it",
)

def retrieve(state: AgentState):
    relevant_docs = retriever.invoke(state["input"])
    return {"context": relevant_docs} if relevant_docs else {"context": []}

def fetch_test_cases(module_name: str):
    module_cases = df_test_cases[df_test_cases["Module name"] == module_name]
    return module_cases.sample(n=min(4, len(module_cases))).to_dict(orient="records") if not module_cases.empty else []

def generate_response(state: AgentState):
    if state["context"] and "solution" in state["context"][0].metadata:
        context_doc = state["context"][0]
        test_cases = fetch_test_cases(context_doc.metadata["module"])

        response_template = """**Error:**\n{Error}\n\n**Solution:**\n{Solution}\n\n**Test Cases:**\n{TestCases}"""

        formatted_cases = "\n\n".join([
            f"**Test Case ID:** {tc['Test Case ID']}\n**Scenario:** {tc['Test Description']}\n**Steps:** {tc['Test Steps']}\n**Expected Result:** {tc['Expected Results']}"
            for tc in test_cases
        ])

        return {"response": response_template.format(
            Error=state["input"],
            Solution=context_doc.metadata["solution"],
            TestCases=formatted_cases if formatted_cases else "No relevant test cases found."
        )}
    return {"response": "**Error:** The defect is unknown and cannot be resolved."}

# Create the workflow graph for the defect resolution agent.
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate_response", generate_response)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate_response")
workflow.add_edge("generate_response", END)
agent_defect = workflow.compile()

def get_solution(defect_description: str) -> str:
    """
    Given a defect description, this function returns the solution and test cases.
    """
    state = AgentState(input=defect_description.strip(), context=[], response="")
    result = agent_defect.invoke(state)
    return result["response"]

# -----------------------------------------------------
# New Agent: LLM-Based Q&A for Additional Issues or Questions
# -----------------------------------------------------

def get_llm_response(query: str) -> str:
    """
    Uses the ChatGroq LLM to generate a response to a general query.
    Note: Pass a string prompt directly.
    """
    prompt = f"Answer the following question as helpfully as possible:\n\nQuestion: {query}"
    result = llm.invoke(prompt)
    return result.content

# -----------------------------------------------------
# Chat Interface: Routing between Defect Agent and LLM Q&A Agent
# -----------------------------------------------------

class ChatAgent:
    def __init__(self):
        # Store the conversation history if needed.
        self.conversation_history = []

    def process_input(self, user_input: str) -> str:
        """
        Processes the user input by routing to the defect resolution agent or the LLM Q&A agent.
        Use the prefix "defect:" for defect descriptions and "question:" (or no prefix) for general questions.
        """
        user_input = user_input.strip()
        # Check for prefix to determine which agent to call.
        if user_input.lower().startswith("defect:"):
            # Remove the prefix and get solution from the defect agent.
            defect_query = user_input[len("defect:"):].strip()
            response = get_solution(defect_query)
        elif user_input.lower().startswith("question:"):
            # Remove the prefix and get answer from the LLM-based Q&A agent.
            query = user_input[len("question:"):].strip()
            response = get_llm_response(query)
        else:
            # If no prefix is provided, assume it's a general query.
            response = get_llm_response(user_input)

        # Optionally, record the conversation history.
        self.conversation_history.append({"role": "user", "message": user_input})
        self.conversation_history.append({"role": "agent", "message": response})
        return response

    def show_history(self):
        """
        Utility to print the conversation history.
        """
        for turn in self.conversation_history:
            print(f"{turn['role'].capitalize()}: {turn['message']}\n")

# -----------------------------------------------------
# Chat Loop Example
# -----------------------------------------------------

if __name__ == "__main__":
    chat_agent = ChatAgent()
    print("Welcome to the Hybrid Defect Resolution & Q&A Chatbot!")
    print("Type 'defect: <your defect description>' to get a solution with test cases.")
    print("Type 'question: <your query>' for general questions (e.g., issues with the solution, clarifications, etc.).")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = chat_agent.process_input(user_input)
        print(f"Agent: {response}\n")

