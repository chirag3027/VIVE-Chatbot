# 📄 Document Q&A Chatbot

A smart chatbot built with Streamlit, LangChain, and OpenAI that can answer questions based on uploaded documents (PDF, DOCX, TXT), show citations, and collect feedback in Google Sheets.

## Features
- Upload multiple document formats
- GPT-4-turbo with fallback to GPT-3.5
- Source-aware responses with expandable citations
- Prompt suggestions for follow-up
- Feedback stored in Google Sheets
- Local logo branding
- Logs every question + answer in "Interactions" tab
- Feedback saved to "Feedback" tab

## Setup

1. Create `.streamlit/secrets.toml` with:
```
OPENAI_API_KEY = "sk-..."
GSERVICE_ACCOUNT = """{...}"""
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run locally:
```
streamlit run app.py
```

4. Deploy via Streamlit Cloud. Set secrets via dashboard.

## Google Sheet Tabs Required:
- **Interactions**: Timestamp | Question | Answer
- **Feedback**: Timestamp | Question | Answer | Feedback

## License
MIT
# VIVE-Chatbot
