from openai import OpenAI
import json
from dotenv import load_dotenv
load_dotenv()
import os

# init client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# store
# chat_history = {}
chat_history = None

# chat function
def chitchat(input_text: str):
    global chat_history

    res = client.responses.create(
        model="gpt-4o-mini",
        instructions="You are ai chatbot that can respond like human. You always respond in the Thai language only.",
        input=input_text,
        previous_response_id=chat_history,
        store=True,
    )
    # update last_id
    chat_history = res.id

    # response
    response_text = res.output_text

    return response_text


if __name__ == "__main__":
    # create chatbot while loop
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        ai_response = chitchat(user_input)
        print(f"AI: {ai_response}\n")