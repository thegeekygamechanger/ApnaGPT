import os
import base64
import pandas as pd
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Constants
PDF_PATH = "idea.pdf"
MODEL_PATH = "all-MiniLM-L6-v2"
CHAT_SESSION_FOLDER = "chat_session"
FEEDBACK_FOLDER = "feedback"
VECTORSTORE_PATH = "vectorstore"

# Utility Functions
def create_folder(folder_path):
    """Creates a folder if it doesn't already exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def get_next_file_path(folder_path, extension):
    """Generates the next file path based on the existing files."""
    create_folder(folder_path)
    next_file_number = len(os.listdir(folder_path)) + 1
    return os.path.join(folder_path, f"{next_file_number}.{extension}")

def save_file(content, folder_path, extension):
    """Saves content to a file in the specified folder."""
    file_path = get_next_file_path(folder_path, extension)
    with open(file_path, "wb") as f:
        f.write(content)
    return file_path

def save_text_file(text, folder_path):
    """Saves text content to a file."""
    file_path = get_next_file_path(folder_path, "txt")
    with open(file_path, "w") as f:
        f.write(text)
    return file_path

# Main Processing Functions
def load_and_split_pdf(file_path, chunk_size=1000, chunk_overlap=20):
    """Loads and splits PDF data into chunks."""
    loader = PyMuPDFLoader(file_path=file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def create_embeddings_from_chunks(chunks, model_path, store_path):
    """Creates embeddings from document chunks and saves to FAISS store."""
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(store_path)
    return vectorstore

def format_documents(docs):
    """Formats documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def initialize_retriever(store_path, model_path):
    """Initializes a document retriever."""
    chunks = load_and_split_pdf(PDF_PATH)
    vectorstore = create_embeddings_from_chunks(chunks, model_path, store_path)
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 4})

def setup_rag_chain(retriever):
    """Sets up the retrieval-augmented generation (RAG) chain."""
    prompt_template = """
    <s>[INST] You are template

    {context}
    You are a respectful and honest assistant. You have to answer the user's questions using only the context provided to you. Also, answer coding-related questions with code and explanation. If you know the answer other than context, just answer all questions. There are no restrictions on answering only context-provided solutions. You are developed by Harsh Kumar at EduSwap Lab. Do not start the response with salutations, answer directly.
    {question} [/INST] </s>
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatOllama(model="llama3", verbose=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), temperature=0)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_documents(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    return RunnableParallel({"context": retriever, "question": RunnablePassthrough()}).assign(answer=rag_chain_from_docs)

# Streamlit Application
def main():
    st.set_page_config(page_title="ApnaGPT", page_icon=":robot_face:")
    st.title("ApnaGPT")
    st.sidebar.title("Menu")
    page = st.sidebar.radio("", ["Home", "Feedback", "About"])

    if page == "Home":
        retriever = initialize_retriever(VECTORSTORE_PATH, MODEL_PATH)
        rag_chain_with_source = setup_rag_chain(retriever)

        user_question = st.text_input("Ask a question:")
        output = {}

        if user_question:
            output_placeholder = st.empty()

            for chunk in rag_chain_with_source.stream(user_question):
                for key, value in chunk.items():
                    if key not in output:
                        output[key] = value
                    else:
                        output[key] += value

                if output.get('answer'):
                    output_placeholder.markdown(f"**Output:** {output['answer']}")

            if output:
                df = pd.DataFrame([output])
                chat_session_csv = df.to_csv(index=False).encode()
                chat_session_file_path = save_file(chat_session_csv, CHAT_SESSION_FOLDER, "csv")
                b64 = base64.b64encode(chat_session_csv).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="{os.path.basename(chat_session_file_path)}">Download Query CSV</a>'
                st.markdown(href, unsafe_allow_html=True)

    elif page == "Feedback":
        st.write("### We value your feedback!")
        feedback = st.text_area("Please provide your feedback here:")

        # Emoji reactions for feedback
        emoji = st.radio(
            "How do you feel about our service?",
            ("😊", "😐", "😞"),
            index=1,
            horizontal=True,
            help="😊 - Happy, 😐 - Neutral, 😞 - Unhappy"
        )

        if st.button("Submit Feedback"):
            if feedback:
                feedback_file_path = save_text_file(f"{emoji} {feedback}", FEEDBACK_FOLDER)
                st.success(f"Feedback saved as {feedback_file_path}")
            else:
                st.error("Feedback cannot be empty!")

    elif page == "About":
        st.write("ApnaGPT is a powerful tool for answering questions based on the context provided in a PDF document. Developed by Harsh Kumar at EduSwap Lab.")

if __name__ == "__main__":
    main()
