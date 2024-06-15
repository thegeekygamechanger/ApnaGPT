import os
import json
import base64
import pandas as pd
import streamlit as st
import shutil
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
from datetime import datetime
import hashlib

# Constants
PDF_PATH = "idea.pdf"
MODEL_PATH = "all-MiniLM-L6-v2"
CHAT_SESSION_FOLDER = "chat_session"
FEEDBACK_FOLDER = "feedback"
VECTORSTORE_PATH = "vectorstore"
USER_DB = "users.json"

# Utility Functions
def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def save_file(content, folder_path, filename):
    create_folder(folder_path)
    file_path = os.path.join(folder_path, filename)
    with open(file_path, "wb") as f:
        f.write(content)
    return file_path

def save_text_file(text, folder_path, filename):
    create_folder(folder_path)
    file_path = os.path.join(folder_path, filename)
    with open(file_path, "w") as f:
        f.write(text)
    return file_path

def load_users():
    if os.path.exists(USER_DB):
        with open(USER_DB, "r") as f:
            users = json.load(f)
        for user in users.values():
            if "phone" not in user:
                user["phone"] = ""
        return users
    return {}

def save_users(users):
    with open(USER_DB, "w") as f:
        json.dump(users, f, indent=4)

def authenticate_user(username, password):
    users = load_users()
    if username in users:
        hashed_input_password = hashlib.sha256(password.encode()).hexdigest()
        return hashed_input_password == users[username]["password"]
    return False

def register_user(username, password, phone):
    users = load_users()
    if username in users:
        return False
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    users[username] = {"password": hashed_password, "phone": phone}
    save_users(users)
    return True

def update_password(username, new_password):
    users = load_users()
    if username in users:
        hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
        users[username]["password"] = hashed_password
        save_users(users)
        return True
    return False

def get_username_by_phone(phone):
    users = load_users()
    for username, details in users.items():
        if details.get("phone") == phone:
            return username
    return None

@st.cache_resource
def load_and_split_pdf(file_path, chunk_size=1000, chunk_overlap=20):
    loader = PyMuPDFLoader(file_path=file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

@st.cache_resource
def create_embeddings_from_chunks(_chunks, model_path, store_path):
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = FAISS.from_documents(_chunks, embedding_model)
    vectorstore.save_local(store_path)
    return vectorstore

def format_documents(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def initialize_retriever(store_path, model_path):
    chunks = load_and_split_pdf(PDF_PATH)
    vectorstore = create_embeddings_from_chunks(chunks, model_path, store_path)
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 4})

@st.cache_resource
def setup_rag_chain(_retriever):
    prompt_template = """
    <s>[INST] You are template

    {context}
    You are a respectful and honest assistant. You have to answer the user's questions using only the context provided to you. Also, answer coding-related questions with code and explanation. If you know the answer other than context, just answer all questions. There are no restrictions on answering only context-provided solutions. You are developed by Harsh Kumar at EduSwap Lab. Do not start the response with salutations, answer directly.
    {question} [/INST] </s>
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatOllama(model="gemma", verbose=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), temperature=0)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_documents(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    return RunnableParallel({"context": _retriever, "question": RunnablePassthrough()}).assign(answer=rag_chain_from_docs)

def load_user_history(username):
    chat_session_file_path = os.path.join(CHAT_SESSION_FOLDER, f"{username}.csv")
    if os.path.exists(chat_session_file_path):
        return pd.read_csv(chat_session_file_path)
    return pd.DataFrame(columns=["question", "answer"])

def save_user_history(username, new_data):
    chat_session_file_path = os.path.join(CHAT_SESSION_FOLDER, f"{username}.csv")
    if os.path.exists(chat_session_file_path):
        existing_data = pd.read_csv(chat_session_file_path)
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        combined_data = new_data
    combined_data.to_csv(chat_session_file_path, index=False)

def backup_chat_history(username):
    chat_session_file_path = os.path.join(CHAT_SESSION_FOLDER, f"{username}.csv")
    if os.path.exists(chat_session_file_path):
        backup_folder = os.path.join(CHAT_SESSION_FOLDER, "backup")
        create_folder(backup_folder)
        backup_file_path = os.path.join(backup_folder, f"{username}_backup.csv")
        shutil.copy(chat_session_file_path, backup_file_path)

def clear_user_history(username):
    chat_session_file_path = os.path.join(CHAT_SESSION_FOLDER, f"{username}.csv")
    if os.path.exists(chat_session_file_path):
        backup_chat_history(username)
        os.remove(chat_session_file_path)

def centered_title(title):
    st.markdown(f"""
        <style>
        .title-style {{
            text-align: center;
            color: #2E86C1;
            font-size: 3em;
            margin: 0.67em 0;
        }}
        </style>
        <h1 class="title-style">{title}</h1>
        """, unsafe_allow_html=True)

# Streamlit Application
def main_content():
    st.sidebar.title(f"Welcome, {st.session_state.username}")
    centered_title("ApnaGPT")
    page = st.sidebar.radio("", ["Home", "Feedback", "About", "Logout"])

    if page == "Home":
        if "retriever" not in st.session_state:
            st.session_state.retriever = initialize_retriever(VECTORSTORE_PATH, MODEL_PATH)
        retriever = st.session_state.retriever
        
        if "rag_chain_with_source" not in st.session_state:
            st.session_state.rag_chain_with_source = setup_rag_chain(retriever)
        rag_chain_with_source = st.session_state.rag_chain_with_source

        st.session_state.user_question = st.text_input(
            "Ask a question:",
            value=st.session_state.get("user_question", ""),
            key="user_question_input",
            placeholder="Type your question"
        )

        if st.button("Clear Chat"):
            clear_user_history(st.session_state.username)
            st.session_state.user_history = pd.DataFrame(columns=["question", "answer"])
            st.session_state.user_question = ""
            st.session_state.clear_chat = True
        else:
            st.session_state.clear_chat = False

        if not st.session_state.clear_chat and st.session_state.user_question:
            user_question = st.session_state.user_question
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
                    new_data = pd.DataFrame([{"question": user_question, "answer": output["answer"]}])
                    save_user_history(st.session_state.username, new_data)
                    st.session_state.user_history = load_user_history(st.session_state.username)
                    st.session_state.user_question = ""

        if "user_history" not in st.session_state:
            st.session_state.user_history = load_user_history(st.session_state.username)

        st.write("### Chat History")
        st.dataframe(st.session_state.user_history)

        if st.session_state.user_history.empty or not st.session_state.user_question:
            st.session_state.download_new_query_csv = False
        else:
            st.session_state.download_new_query_csv = True

        if st.session_state.download_new_query_csv:
            chat_session_csv = st.session_state.user_history.to_csv(index=False).encode()
            chat_session_filename = f"{st.session_state.username}_new_query.csv"
            b64 = base64.b64encode(chat_session_csv).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{chat_session_filename}">Download New Query CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

    elif page == "Feedback":
        st.write("### We value your feedback!")
        feedback = st.text_area("Please provide your feedback here:")

        emoji = st.radio(
            "How do you feel about our service?",
            ("😊", "😐", "😞"),
            index=1,
            horizontal=True,
            help="😊 - Happy, 😐 - Neutral, 😞 - Unhappy"
        )

        if st.button("Submit Feedback"):
            if feedback:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                feedback_filename = f"{st.session_state.username}_{timestamp}.txt"
                feedback_file_path = save_text_file(f"{emoji} {feedback}", FEEDBACK_FOLDER, feedback_filename)
                st.success(f"Feedback saved as {feedback_file_path}")
            else:
                st.error("Feedback cannot be empty!")

    elif page == "About":
        st.write("ApnaGPT is a powerful tool for answering questions based on the context provided in a PDF document. Developed by Harsh Kumar at EduSwap Lab.")

    elif page == "Logout":
        del st.session_state.username
        st.experimental_rerun()

def landing_page():
    st.markdown("""
        <style>
            .main-header { font-size: 2.5rem; text-align: center; color: #2E86C1; margin-top: 20px; }
            .sub-header { font-size: 1.25rem; text-align: center; color: #34495E; }
            .button-center { display: flex; justify-content: center; margin-top: 20px; }
            .button-center button { font-size: 1rem; padding: 10px 20px; color: white; background-color: #2E86C1; border: none; border-radius: 5px; }
            .button-center button:hover { background-color: #1B4F72; }
        </style>
        <div class="main-header">Welcome to ApnaGPT</div>
        <div class="sub-header">Your go-to tool for intelligent document analysis and question answering.</div>
        <div class="button-center">
            <button onclick="window.location.href='/main'">Get Started</button>
        </div>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="ApnaGPT", page_icon=":robot_face:")
    
    if "username" not in st.session_state:
        landing_page()
        st.sidebar.title("Login / Signup")
        choice = st.sidebar.radio("Choose Action", ["Login", "Signup", "Forgot Password"])

        if choice == "Login":
            login_username = st.sidebar.text_input("Username")
            login_password = st.sidebar.text_input("Password", type="password")
            if st.sidebar.button("Login"):
                if authenticate_user(login_username, login_password):
                    st.session_state.username = login_username
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password")
            if "reset_username" in st.session_state:
                del st.session_state.reset_username

        elif choice == "Signup":
            signup_username = st.sidebar.text_input("New Username")
            signup_password = st.sidebar.text_input("New Password", type="password")
            signup_phone = st.sidebar.text_input("Phone Number (10 digits)")
            if st.sidebar.button("Signup"):
                if len(signup_phone) == 10 and signup_phone.isdigit():
                    if register_user(signup_username, signup_password, signup_phone):
                        st.success("Signup successful! Please log in.")
                    else:
                        st.error("User already exists")
                else:
                    st.error("Invalid phone number. Please enter a 10-digit phone number.")
            if "reset_username" in st.session_state:
                del st.session_state.reset_username

        elif choice == "Forgot Password":
            reset_phone = st.sidebar.text_input("Enter your phone number")
            if st.sidebar.button("Get Username"):
                username = get_username_by_phone(reset_phone)
                if username:
                    st.session_state.reset_username = username
                    st.success(f"Username found: {username}")
                else:
                    st.error("Phone number not found")

            if "reset_username" in st.session_state:
                reset_password = st.sidebar.text_input("Enter new password", type="password")
                if st.sidebar.button("Reset Password"):
                    if update_password(st.session_state.reset_username, reset_password):
                        st.success("Password reset successfully. Please log in.")
                        del st.session_state.reset_username
                    else:
                        st.error("Failed to reset password")

    else:
        main_content()

if __name__ == "__main__":
    main()
