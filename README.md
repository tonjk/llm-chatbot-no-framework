# llm-chatbot-no-framework
AI Chatbot without using framework

## 🚀 Update: OpenAI ChatGPT Responses API Integration

This project now uses the **OpenAI ChatGPT Responses API** via the `openai` Python SDK (`client.responses.create(...)`) to build a simple CLI chatbot **without any external chatbot framework**.

### Why this update (Chat Completions → Responses API)
- **One API for modern “chat + tools”**: the Responses API is the current interface for conversational generation and tool-augmented flows.
- **Conversation state**: you can keep context by passing `previous_response_id`, instead of manually managing message arrays.
- **Cleaner output**: the SDK provides convenience fields like `response.output_text` for the common “just give me the assistant text” case.

### What’s in this repo
- **`src/openai_chatbot.py`**: minimal Thai-only CLI chatbot using `responses.create` and `previous_response_id`.
- **`src/openai_chatbot_tools.py`**: an “agentic” demo showing how to wire tools (built-in `web_search` + a simulated custom RAG function).

### Requirements
- Python (see `.python-version`)
- An OpenAI API key in your environment as `OPENAI_API_KEY`

### Setup
1) Create and activate a virtual environment.

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Set your API key (example):

```bash
export OPENAI_API_KEY="YOUR_KEY_HERE"
```

### Run
Minimal chatbot:

```bash
python src/openai_chatbot.py
```

Tools/agent demo:

```bash
python src/openai_chatbot_tools.py
```

### Notes
- The minimal chatbot is configured to **respond in Thai only** (see `instructions` in `src/openai_chatbot.py`).
- For multi-turn chat, the script stores the latest response id and sends it back as `previous_response_id` on the next turn.