# LLMs

def retrieve(state: AgentState):
    try:
        error_message = state["input"].strip().lower()
        
        # Check for an exact match in the CSV
        exact_match = df[df["Description"].str.lower() == error_message]
        
        if not exact_match.empty:
            # Convert exact match to a Document for consistency
            matched_row = exact_match.iloc[0]
            matched_doc = Document(
                page_content=matched_row["Description"],
                metadata={
                    "solution": matched_row["Solution"],
                    "module": matched_row["Module"]
                }
            )
            return {"context": [matched_doc]}
        
        # If no exact match, return no match found
        return {"context": [], "response": "**Error**: No matching defect found in the database."}
    
    except Exception as e:
        return {"context": [], "response": f"Error processing request: {str(e)}"}
