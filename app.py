# Static Configuration Variables
PAGE_TITLE = "Document Q&A Bot"  # Page title for the Streamlit app
LOGO_FILENAME = "logo.png"  # Filename for the logo image to display
SPREADSHEET_NAME = "Chatbot Feedback"  # Google Sheets document name
INTERACTIONS_SHEET_NAME = "Interactions"  # Sheet name for storing Q&A pairs
FEEDBACK_SHEET_NAME = "Feedback"  # Sheet name for storing user feedback
CHUNK_SIZE = 1000  # Character chunk size for splitting documents
CHUNK_OVERLAP = 100  # Character overlap between chunks
SEARCH_K = 5  # Number of relevant documents to retrieve
MODEL_NAME_PRIMARY = "gpt-4-turbo"  # Preferred OpenAI model
MODEL_NAME_FALLBACK = "gpt-3.5-turbo"  # Fallback OpenAI model
MODEL_TEMPERATURE = 0  # Temperature setting for deterministic output
DEFAULT_PROMPT = "You are a helpful assistant for sales agents. Simplify answers, provide examples and speak in clear language."
NO_ANSWER_RESPONSE = "No answer found in the uploaded docs, please reach out to 'apoorv.kamra@gmail.com'"

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
import tempfile
import os
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from openai.error import AuthenticationError, InvalidRequestError
from datetime import datetime
import traceback

def initialize_app():
    try:
        st.set_page_config(page_title=PAGE_TITLE, layout="wide")
        st.title("📄 Chat with Your Documents")
        logo_path = os.path.join(os.path.dirname(__file__), LOGO_FILENAME)
        if os.path.exists(logo_path):
            st.image(logo_path, width=120)
    except Exception as e:
        st.error("Failed to initialize the app.")
        st.exception(e)

def get_api_key():
    try:
        # First, try to get the key from the secrets file
        api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            # If not found in secrets, prompt the user
            api_key = st.text_input("Enter your OpenAI API Key", type="password")
        if not api_key:
            st.warning("API Key is required to proceed.")
            st.stop()
        return api_key
    except Exception as e:
        st.error("Failed to retrieve OpenAI API key.")
        st.exception(e)
        st.stop()

@st.cache_resource(show_spinner=False)
def load_llm(api_key):
    try:
        return ChatOpenAI(model_name=MODEL_NAME_PRIMARY, temperature=MODEL_TEMPERATURE, openai_api_key=api_key)
    except Exception as e:
        st.warning("Falling back to " + MODEL_NAME_FALLBACK)
        try:
            return ChatOpenAI(model_name=MODEL_NAME_FALLBACK, temperature=MODEL_TEMPERATURE, openai_api_key=api_key)
        except Exception as fallback_error:
            st.error("Failed to load any language model.")
            st.exception(fallback_error)
            st.stop()

@st.cache_resource(show_spinner=False)
def initialize_sheets():
    try:
        creds_dict = json.loads(st.secrets["GSERVICE_ACCOUNT"])
        scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
        credentials = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        gc = gspread.authorize(credentials)
        sheet = gc.open(SPREADSHEET_NAME)
        return {
            "interactions": sheet.worksheet(INTERACTIONS_SHEET_NAME) if INTERACTIONS_SHEET_NAME in [ws.title for ws in sheet.worksheets()] else sheet.add_worksheet(title=INTERACTIONS_SHEET_NAME, rows="1000", cols="3"),
            "feedback": sheet.worksheet(FEEDBACK_SHEET_NAME) if FEEDBACK_SHEET_NAME in [ws.title for ws in sheet.worksheets()] else sheet.add_worksheet(title=FEEDBACK_SHEET_NAME, rows="1000", cols="4")
        }
    except Exception as e:
        st.error("Google Sheet not connected.")
        st.exception(e)
        return None

def prepare_sheets(sheets):
    try:
        if sheets["interactions"].row_count == 1:
            sheets["interactions"].append_row(["Timestamp", "Question", "Answer"])
        if sheets["feedback"].row_count == 1:
            sheets["feedback"].append_row(["Timestamp", "Question", "Answer", "Feedback"])
    except Exception as e:
        st.warning("Error initializing Google Sheet rows.")
        st.exception(e)

def process_documents(uploaded_files):
    documents = []
    try:
        for uploaded_file in uploaded_files:
            path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.read())
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif uploaded_file.name.endswith(".txt"):
                loader = TextLoader(path)
            elif uploaded_file.name.endswith(".docx"):
                loader = Docx2txtLoader(path)
            else:
                st.warning(f"Unsupported format: {uploaded_file.name}")
                continue
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["file_path"] = uploaded_file.name
                doc.metadata["file_url"] = path
                doc.metadata["source"] = f"Page {doc.metadata.get('page', loaded_docs.index(doc)+1)}"
            documents.extend(loaded_docs)
    except Exception as e:
        st.error("Failed to process documents.")
        st.exception(e)
    return documents

def build_qa_chain(documents, api_key, llm):
    try:
        splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        texts = splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        db = FAISS.from_documents(texts, embeddings)
        retriever = db.as_retriever(search_kwargs={"k": SEARCH_K})
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        return ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, return_source_documents=True)
    except Exception as e:
        st.error("Failed to build QA chain.")
        st.exception(e)
        st.stop()

def render_sources(sources):
    seen = set()
    for i, doc in enumerate(sources):
        try:
            meta = doc.metadata
            file = meta.get("file_path", "Document")
            page = meta.get("page", i+1)
            key = f"{file}-{page}"
            if key in seen: continue
            seen.add(key)
            snippet = doc.page_content.strip()[:500]
            st.markdown(f"**📄 {file}, Page {page}**")
            st.code(snippet + "...", language="text")
        except Exception as e:
            st.warning("Error rendering a document source.")
            st.exception(e)

# Main App Execution
initialize_app()
openai_api_key = get_api_key()
llm = load_llm(openai_api_key)
sheets = initialize_sheets()
if sheets:
    prepare_sheets(sheets)

uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True)

if uploaded_files:
    docs = process_documents(uploaded_files)
    qa_chain = build_qa_chain(docs, openai_api_key, llm)

    user_input = st.chat_input("Ask a question about your documents")
    if user_input:
        try:
            full_input = f"{DEFAULT_PROMPT}\n\nQuestion: {user_input}"
            result = qa_chain({"question": full_input, "chat_history": st.session_state.get("chat_history", [])})
            st.chat_message("user").write(user_input)

            answer = result.get("answer", "").strip()
            sources = result.get("source_documents", [])
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if answer and sources:
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    if sheets: sheets["interactions"].append_row([timestamp, user_input, answer])
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("👍 Good answer", key=user_input+"_up"):
                            if sheets: sheets["feedback"].append_row([timestamp, user_input, answer, "👍"])
                    with col2:
                        if st.button("👎 Needs improvement", key=user_input+"_down"):
                            if sheets: sheets["feedback"].append_row([timestamp, user_input, answer, "👎"])
                    st.markdown("**💡 Try a follow-up:**")
                    st.markdown("- Give more details")
                    st.markdown("- Share data points from the text")
                    st.markdown("- Provide an example")
                    with st.expander("📚 View sources for this answer"):
                        render_sources(sources)
            else:
                st.chat_message("assistant").write(NO_ANSWER_RESPONSE)
                if sheets: sheets["interactions"].append_row([timestamp, user_input, NO_ANSWER_RESPONSE])
        except Exception as e:
            st.error("An error occurred while generating the answer.")
            st.exception(e)
