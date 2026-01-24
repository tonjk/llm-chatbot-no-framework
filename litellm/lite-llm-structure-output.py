import os
import litellm
from litellm import completion 
from pydantic import BaseModel


# response = completion(model="gpt-4o-mini",
#                       response_format={"type": "json_object"},
#                       messages=[
#                           {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
#                           {"role": "user", "content": "Who won the world series in 2020?"},
#                           ]
#                           )

# print(response.choices[0].message.content)
# {
#   "year": 2020,
#   "winner": "Los Angeles Dodgers",
#   "opponent": "Tampa Bay Rays",
#   "series_result": "Los Angeles Dodgers won 4-2"
# }

# ------------------------------------------------------------------------------------------

litellm.enable_json_schema_validation=True # LiteLLM will validate the json response using jsonvalidator
messages = [{"role": "user", "content": "List 5 important events in the XIX century"}]

class CalendarEvent(BaseModel):
  name: str
  date: str
  participants: list[str]

class EventsList(BaseModel):
    events: list[CalendarEvent]

resp = completion(
    model="gpt-4o-2024-08-06",
    messages=messages,
    response_format=EventsList
)

print("Received={}".format(resp))

events_list = EventsList.model_validate_json(resp.choices[0].message.content)
print(events_list)

# ------------------------------------------------------------------------------------------
# litellm._turn_on_debug()
# # gemini 2.0+
# class UserInfo(BaseModel):
#     name: str
#     age: int

# response = completion(
#     model="gemini/gemini-2.0-flash",
#     messages=[{"role": "user", "content": "Extract: John is 25 years old"}],
#     response_format={
#         "type": "json_schema",
#         "json_schema": {
#             "name": "user_info",
#             "schema": {
#                 "type": "object",
#                 "properties": {
#                     "name": {"type": "string"},
#                     "age": {"type": "integer"}
#                 },
#                 "required": ["name", "age"],
#                 "additionalProperties": False  # Supported on Gemini 2.0+
#             }
#         }
#     }
# )

# print(response)