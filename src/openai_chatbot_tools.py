import json
from openai import OpenAI
from typing import List, Dict

# Ensure you have your API key set in your environment variables
from dotenv import load_dotenv
load_dotenv()
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 1. Define the Custom RAG Tool (Simulated) ---
def query_knowledge_base(query: str) -> str:
    """
    Simulates a RAG (Retrieval Augmented Generation) lookup.
    In production, this would query a vector DB like Pinecone or Chroma.
    """
    # print(f"\n[System] 🔍 RAG Tool Triggered: Searching for '{query}'...")
    
    # Mock knowledge base data
    knowledge_base = {
        "refund policy": "Our refund policy allows returns within 30 days of purchase. Items must be unworn.",
        "api rate limit": "The standard API rate limit is 5000 requests per minute for enterprise accounts.",
        "company hours": "Support hours are 9 AM to 5 PM PST, Monday through Friday."
    }
    
    # Simple keyword matching for simulation
    results = []
    for topic, content in knowledge_base.items():
        if topic in query.lower():
            results.append(content)
            
    if results:
        return "\n".join(results)
    return "No specific internal documents found for this query."

# Tool definition schema for the custom RAG function
rag_tool_definition = {
    "type": "function",
    "name": "query_knowledge_base",
    "description": "Look up internal company policies, technical docs, or specific business data.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up in the internal database."
            }
        },
        "required": ["query"]
    }
}

# --- 2. The Chatbot Agent ---
def run_agentic_chat():
    print("--- Advanced AI Agent (Responses API) ---")
    print("Capabilities: 🌐 Web Search | 📂 Internal RAG")
    print("Type 'exit' to quit.\n")

    last_response_id = None

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        try:
            # The Responses API call
            # Note: We pass both the built-in 'web_search' and our custom tool.
            response = client.responses.create(
                model="gpt-4o-mini", 
                input=user_input,
                previous_response_id=last_response_id, # Maintains conversation state automatically
                tools=[
                    {"type": "web_search"},  # Built-in tool
                    rag_tool_definition      # Custom RAG tool
                ]
            )

            # --- 3. Handle Tool Execution (If the model decides to call custom tools) ---
            # The Responses API might execute built-in tools (like web_search) automatically,
            # but custom functions often require client-side execution if not hosted.
            # However, the modern Responses API often returns the *intent* to call items.
            
            final_output_text = ""
            
            # Iterate through the output items
            if hasattr(response, 'output'):
                for item in response.output:
                    
                    # Case A: The model generated text directly
                    if item.type == "message" and hasattr(item, 'content'):
                        final_output_text += item.content

                    # Case B: The model wants to call our custom RAG function
                    elif item.type == "function_call":
                        if item.name == "query_knowledge_base":
                            args = json.loads(item.arguments)
                            tool_result = query_knowledge_base(args["query"])
                            
                            # In the Responses API, we can submit the tool result back 
                            # to generate the final answer in a new turn, or if using 
                            # server-side execution, it might handle it. 
                            # Here, we simulate the agent receiving the data.
                            
                            # Create a follow-up response with the tool output
                            follow_up = client.responses.create(
                                model="gpt-4o",
                                input=f"Tool Output: {tool_result}",
                                previous_response_id=response.id
                            )
                            # Update reference to the new response
                            response = follow_up
                            # Extract text from the follow-up
                            final_output_text = follow_up.output[0].content

            # Store ID for the next turn to maintain context
            last_response_id = response.id
            
            # Print the final response
            # (If the API returned a list of items, we join the text parts)
            if not final_output_text:
                # Fallback extraction depending on exact API version response shape
                final_output_text = " ".join([
                    i.content for i in response.output if i.type == 'message'
                ])
            
            print(f"AI: {final_output_text}\n")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_agentic_chat()