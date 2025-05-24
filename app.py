###############################################################################
#                          VIVE MedInfo Chatbot Demo                          #
###############################################################################

# -------------------------------
# Static Configuration Variables
# -------------------------------
PAGE_TITLE = "Document Q&A Bot"
LOGO_FILENAME = "VIVE.jpeg"
TITLE = "Vive MedInfo Chatbot Demo"
LOGO_WIDTH = 120

SPREADSHEET_NAME = "Chatbot Feedback"
INTERACTIONS_SHEET_NAME = "Interactions"
FEEDBACK_SHEET_NAME = "Feedback"
DEFAULT_SHEET_ROWS = 1000
DEFAULT_INTERACTIONS_COLS = 3
DEFAULT_FEEDBACK_COLS = 4

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
SEARCH_K = 5
MODEL_NAME_PRIMARY = "gpt-4-turbo"
MODEL_NAME_FALLBACK = "gpt-3.5-turbo"
MODEL_TEMPERATURE = 0

DEFAULT_PROMPT = (
    "You are a helpful assistant for a pharmaceutical sales and medical affairs team that frequently engages with doctors and healthcare professionals. "
    "Respond in a simple, clear way. Use friendly and precise language appropriate for healthcare professionals based in USA and Canada."
    "Refer only to the information provided in uploaded document and if not found then return an empty string"
)

NO_ANSWER_RESPONSE = (
    "No answer found in the uploaded docs, please reach out to 'apoorv.kamra@gmail.com'"
)

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
MIN_WORD_COUNT = 6  # Ensure responses are substantive

###############################################################################
#                                Imports                                      #
###############################################################################
import json
import os
import re
import tempfile
from datetime import datetime

import gspread
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from oauth2client.service_account import ServiceAccountCredentials
import io
# # PDF generation (optional)
# try:
#     from fpdf import FPDF
#     PDF_AVAILABLE = True
# except ImportError:
#     PDF_AVAILABLE = False
    # PDF functionality will be disabled until fpdf is installed

###############################################################################
#                        Streamlit Configuration                              #
###############################################################################
st.set_page_config(page_title=PAGE_TITLE, layout="wide")

###############################################################################
#                              Sidebar Settings                               #
###############################################################################

###############################################################################
#                          Session State Initialization                       #
###############################################################################
# Ensure chat history and follow-up prompt keys are present in session state
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("follow_prompt", "")

###############################################################################
#                            Helper Functions                                #
###############################################################################

def initialize_app():
    """
    Render the application header: display the logo (if available) and main title.
    Called once when the app loads.
    """
    logo_path = os.path.join(os.path.dirname(__file__), LOGO_FILENAME)
    if os.path.exists(logo_path):
        # Display centered logo at specified width
        st.image(logo_path, width=LOGO_WIDTH)
    # Show the main title of the chatbot
    st.title(TITLE)


@st.cache_resource(show_spinner=False)
def load_llm(api_key: str) -> ChatOpenAI:
    """
    Initialize and cache the OpenAI chat model.
    Attempts to load the primary model first; on failure, falls back.

    Args:
        api_key: Your OpenAI API key.

    Returns:
        An instance of ChatOpenAI ready for generating responses.
    """
    try:
        return ChatOpenAI(model=MODEL_NAME_PRIMARY, temperature=MODEL_TEMPERATURE, api_key=api_key)
    except Exception:
        st.warning(f"Primary model failed, falling back to {MODEL_NAME_FALLBACK}")
        return ChatOpenAI(model=MODEL_NAME_FALLBACK, temperature=MODEL_TEMPERATURE, api_key=api_key)


def get_api_key() -> str:
    """
    Retrieve the OpenAI API key from Streamlit secrets or via user input.
    Stops execution if no key is provided.

    Returns:
        The OpenAI API key as a string.
    """
    key = os.environ.get("OPENAI_API_KEY")
    # 2) Fallback to Streamlit secrets.toml (for Streamlit Cloud)
    if not key:
        key = st.secrets.get("OPENAI_API_KEY")
    #or st.text_input("OpenAI API Key", type="password")
    if not key:
        # Prompt user that the key is mandatory
        st.warning("API Key is required to proceed.")
        st.stop()
    return key


@st.cache_resource(show_spinner=False)
def init_sheets():
    """
    Authenticate and connect to Google Sheets, ensuring worksheets exist.

    Returns:
        A dict with 'interactions' and 'feedback' worksheet handles.
    """
    creds_dict = json.loads(st.secrets["GSERVICE_ACCOUNT"], strict=False)
    scope = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive',
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    sheet = client.open(SPREADSHEET_NAME)

    def get_or_create(title, rows, cols):
        # Return existing worksheet or create a new one
        if title in [ws.title for ws in sheet.worksheets()]:
            return sheet.worksheet(title)
        return sheet.add_worksheet(title=title, rows=rows, cols=cols)

    return {
        "interactions": get_or_create(INTERACTIONS_SHEET_NAME, DEFAULT_SHEET_ROWS, DEFAULT_INTERACTIONS_COLS),
        "feedback": get_or_create(FEEDBACK_SHEET_NAME, DEFAULT_SHEET_ROWS, DEFAULT_FEEDBACK_COLS),
    }


def prepare_sheets(worksheets: dict):
    """
    If worksheets are new (only 1 row), append header rows.

    Args:
        worksheets: Dict containing worksheet handles.
    """
    if worksheets["interactions"].row_count == 1:
        worksheets["interactions"].append_row(["Timestamp", "Question", "Answer"])
    if worksheets["feedback"].row_count == 1:
        worksheets["feedback"].append_row(["Timestamp", "Question", "Answer", "Feedback"])


def process_documents(files: list) -> list:
    """
    Load and parse uploaded documents, attaching metadata for tracing.
    Supports PDF, TXT, and DOCX formats.

    Args:
        files: List of uploaded file objects from Streamlit.

    Returns:
        List of LangChain Document objects with metadata.
    """
    docs = []
    for f in files:
        tmp_path = os.path.join(tempfile.gettempdir(), f.name)
        with open(tmp_path, "wb") as out:
            out.write(f.read())

        # Select appropriate loader based on extension
        if f.name.endswith('.pdf'):
            loader = PyPDFLoader(tmp_path)
        elif f.name.endswith('.txt'):
            loader = TextLoader(tmp_path)
        elif f.name.endswith('.docx'):
            loader = Docx2txtLoader(tmp_path)
        else:
            st.warning(f"Unsupported file type: {f.name}")
            continue

        # Load pages/chunks and annotate metadata
        for doc in loader.load():
            doc.metadata.update({
                "file_path": f.name,
                "file_url": tmp_path,
                "source": f"Page {doc.metadata.get('page', 0) + 1}",
            })
            docs.append(doc)
    return docs

###############################################################################
#                       Custom Prompt & QA Chain Builder                      #
###############################################################################
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Use ONLY the following context to answer. If not in context, respond with '{no_answer}'.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    ).replace('{no_answer}', NO_ANSWER_RESPONSE),
)

def build_qa_chain(docs, api_key, llm):
    # Split documents into overlapping chunks for retrieval
    splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = splitter.split_documents(docs)

    # Create or load FAISS index
    db = FAISS.from_documents(texts, OpenAIEmbeddings(api_key=api_key))

    # Build the conversational chain with memory and custom prompt
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=db.as_retriever(search_kwargs={"k": SEARCH_K}),
        memory=ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"    # ← specify which output to store in memory
        ),
        return_source_documents=True,
        output_key="answer",
    )
    return chain, db

###############################################################################
#                               Source Renderer                               #
###############################################################################

def render_sources(dss: list, query: str):
    """
    Display retrieved document snippets without relevance scores.
    Highlights the matched query term and avoids duplicates.

    Args:
        dss: List of Document objects.
        query: The user’s original question for highlighting.
    """
    seen = set()
    pattern = re.compile(rf"([^.]*{re.escape(query)}[^.]*\.)", re.IGNORECASE)

    for doc in dss:
        key = f"{doc.metadata['file_path']}-{doc.metadata['source']}"
        if key in seen:
            continue
        seen.add(key)

        text = doc.page_content
        match = pattern.search(text)
        snippet = match.group(1) if match else text[:200] + "..."
        highlighted = pattern.sub(lambda m: f"<mark>{m.group(1)}</mark>", snippet)

        # Render file and source (without score)
        st.markdown(f"**📄 {doc.metadata['file_path']}, {doc.metadata['source']}**", unsafe_allow_html=True)
        st.markdown(highlighted, unsafe_allow_html=True)
###############################################################################
#                            Main Execution Flow                             #
###############################################################################

# 1) Render header with logo and title
initialize_app()

# 2) Reset button: clear history and restart
if st.button("Reset Chat"):
    st.session_state.clear()
    st.rerun()

# 3) Retrieve API key and initialize services
api_key = get_api_key()
llm = load_llm(api_key)
sheets = init_sheets()
prepare_sheets(sheets)

# 4) Document upload and processing
uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True)
if uploaded_files:
    docs = process_documents(uploaded_files)
    st.success("✅ Documents processed. Ask a question below.")

    # Build or reuse the QA chain and FAISS index
    qa_chain, db = build_qa_chain(docs, api_key, llm)

    # 5) Display previous chat history
    for msg in st.session_state["chat_history"]:
        st.chat_message(msg['role']).write(msg['message'])

    # 6) Input: user question, possibly prefilled for follow-ups
    if st.session_state.get("follow_prompt"):
        st.session_state["chat_input"] = st.session_state.pop("follow_prompt")
    user_input = st.chat_input("Ask a question", key="chat_input")

    if user_input:
        # Record user question in history
        st.session_state['chat_history'].append({'role': 'user', 'message': user_input})
        st.chat_message('user').write(user_input)

        # 7) Generate answer using the chain
        result = qa_chain({'question': f"{DEFAULT_PROMPT}\n\n{user_input}", 'chat_history': st.session_state['chat_history']})
        answer = result.get('answer', '').strip()
        source_docs = result.get('source_documents', [])
        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)

        # 8) Validate that answer is grounded in documents
        valid = (
            bool(source_docs)
            and answer
            and len(answer.split()) >= MIN_WORD_COUNT
            and answer != NO_ANSWER_RESPONSE
        )

        if valid:
            # Append and render assistant answer
            st.session_state['chat_history'].append({'role': 'assistant', 'message': answer})
            with st.chat_message('assistant'):
                st.markdown(answer)
                # Log interaction to Google Sheets
                sheets['interactions'].append_row([timestamp, user_input, answer])

                # Feedback and follow-up suggestion buttons
                cols = st.columns(2)
                if cols[0].button('👍 Good answer'):
                    sheets['feedback'].append_row([timestamp, user_input, answer, '👍'])
                if cols[1].button('👎 Needs improvement'):
                    sheets['feedback'].append_row([timestamp, user_input, answer, '👎'])
                

                # 9) Collapsible section to view sources
                with st.expander('📚 Sources', expanded=False):
                    render_sources(source_docs, user_input)
        else:
            # No valid context: fallback message
            st.session_state['chat_history'].append({'role': 'assistant', 'message': NO_ANSWER_RESPONSE})
            st.chat_message('assistant').write(NO_ANSWER_RESPONSE)
            sheets['interactions'].append_row([timestamp, user_input, NO_ANSWER_RESPONSE])
