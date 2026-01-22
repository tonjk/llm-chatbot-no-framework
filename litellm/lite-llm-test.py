from litellm import completion
import os
from dotenv import load_dotenv
load_dotenv()


response = completion(
  model="openai/gpt-4o",
  messages=[{ "content": "Hello, how are you?","role": "user"}]
)

print(response)  # The response object contains the model's reply and metadata
print("---" * 20) 
print(response.choices[0].message.content)  # The actual text response from the model