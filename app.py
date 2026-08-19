import os
import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from dotenv import load_dotenv

load_dotenv()

DB_PATH = ".chroma_db"

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
    <style>
        .block-container {
            padding-top: 3rem;
        }
        footer {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag_backend():

    api_key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )

    if not api_key:
        st.error(
            "Missing GOOGLE_API_KEY/GEMINI_API_KEY. "
            "Please add your Gemini API key in Streamlit Secrets."
        )
        return None, None

    # Make the key available to LangChain Google GenAI
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["GEMINI_API_KEY"] = api_key

    if not os.path.exists(DB_PATH):
        st.error(
            "Persistent vector directory not found. "
            "Please make sure .chroma_db exists."
        )
        return None, None

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key
        )

        vector_store = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings
        )

        retriever = vector_store.as_retriever(
            search_kwargs={"k": 6}
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=api_key
        )

        return retriever, llm

    except Exception as e:
        st.error(f"Failed to initialize RAG backend: {e}")
        return None, None


retriever, llm = initialize_rag_backend()


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content":
            "Hello! I've loaded your document index. "
            "What would you like to know?"
        }
    ]


st.title("📄 Document Knowledge Base")
st.caption("Retrieval-Augmented Generation Chatbot powered by Gemini")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


if user_query := st.chat_input(
    "Ask a question about your documents..."
):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    with st.chat_message("user"):
        st.write(user_query)

    if retriever and llm:

        with st.chat_message("assistant"):

            response_placeholder = st.empty()

            with st.spinner(
                "Searching document context..."
            ):

                try:

                    matched_docs = retriever.invoke(
                        user_query
                    )

                    context_text = "\n\n".join(
                        [
                            doc.page_content
                            for doc in matched_docs
                        ]
                    )

                    prompt_template = f"""
You are a precise document analysis assistant.

Answer the question based strictly on the provided context.

If the answer is not present in the context,
say that you don't know.

--- CONTEXT ---
{context_text}
---------------

Question:
{user_query}

Answer:
"""

                    execution_result = llm.invoke(
                        prompt_template
                    )

                    output_text = execution_result.content

                    response_placeholder.write(
                        output_text
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": output_text
                        }
                    )

                except Exception as e:

                    error_msg = (
                        f"An execution error occurred: {str(e)}"
                    )

                    response_placeholder.write(
                        error_msg
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_msg
                        }
                    )

    else:

        st.warning(
            "Backend pipeline initialization failed. "
            "Check your Gemini API credentials."
        )
