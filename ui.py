import streamlit as st
import pandas as pd
from datetime import datetime
import base64
import os
from authenticator import Authenticator
from file_operations import save_text_file
from retriever_setup import initialize_retriever, format_documents
from langchain.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config import CHAT_SESSION_FOLDER, FEEDBACK_FOLDER, VECTORSTORE_PATH, MODEL_PATH

# ChatManager class
class ChatManager:
    @staticmethod
    def load_user_history(username):
        filepath = os.path.join(CHAT_SESSION_FOLDER, f"{username}_history.csv")
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            return pd.DataFrame(columns=["timestamp", "question", "answer"])

    @staticmethod
    def save_user_history(username, new_data):
        filepath = os.path.join(CHAT_SESSION_FOLDER, f"{username}_history.csv")
        existing_data = pd.DataFrame()
        if os.path.exists(filepath):
            existing_data = pd.read_csv(filepath)
        
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        combined_data.to_csv(filepath, index=False)

    @staticmethod
    def clear_user_history(username):
        filepath = os.path.join(CHAT_SESSION_FOLDER, f"{username}_history.csv")
        if os.path.exists(filepath):
            os.remove(filepath)

# Functions for RAG chain setup and backup
def setup_rag_chain(_retriever):
    prompt_template = """
    <s>[INST] You are a template assistant

    {context}
    You are a respectful and honest assistant. Answer the user's questions using only the context provided. Also, answer coding-related questions with code and explanation. If you know the answer other than context, just answer all questions. Do not start the response with salutations, answer directly.
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

    return RunnableParallel({"context": _retriever, "question": RunnablePassthrough()}).assign(answer=rag_chain_from_docs)

def backup_chat_history(username):
    backup_folder = os.path.join(FEEDBACK_FOLDER, "backup")
    os.makedirs(backup_folder, exist_ok=True)
    backup_filename = os.path.join(backup_folder, f"{username}_chat_backup_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv")
    st.session_state.user_history.to_csv(backup_filename, index=False)

# Main content function
def main_content(username):
    st.sidebar.image("/home/harshk/LLM/modular/c.png", width=100)  # Replace with your image URL and width
    st.sidebar.markdown(f'<h1 style="color: Navy;">Welcome, {username}</h1>', unsafe_allow_html=True)
    page = st.sidebar.radio("", ["Home", "Feedback", "About", "Logout"])

    # Initialize session state for user history
    if "user_history" not in st.session_state or st.session_state.get("current_username") != username:
        st.session_state.user_history = ChatManager.load_user_history(username)
        st.session_state.current_username = username

    if page == "Home":
    #     st.markdown("""
    #          #               <style>
    #         .fade-in {
    #             animation: fadeIn 1s ease-in-out forwards;
    #             color: #3498DB; /* Blue color */
    #             text-align: center; /* Center align text */
    #             opacity: 0; /* Start with opacity 0 for fade-in effect */
    #         }
    #         @keyframes fadeIn {
    #             from { opacity: 0; }
    #             to { opacity: 1; }
    #         }
    #     </style>
    #    <h1> <div class="fade-in">
    #          ApnaGPT 
    #     </div></h1>
    #         """, unsafe_allow_html=True)

        if "retriever" not in st.session_state:
            st.session_state.retriever = initialize_retriever(VECTORSTORE_PATH, MODEL_PATH)
        retriever = st.session_state.retriever

        if "rag_chain_with_source" not in st.session_state:
            st.session_state.rag_chain_with_source = setup_rag_chain(retriever)
        rag_chain_with_source = st.session_state.rag_chain_with_source

        # User Input Section (at the top)
        with st.form(key='user_input_form', clear_on_submit=True):
            user_question = st.text_input("Ask your question:", key='user_question_input', placeholder="Write Prompt")
            submit_button_clicked = st.form_submit_button("Submit")

            if submit_button_clicked:
                st.session_state.user_question = user_question  # Update session state with the question
                output = {}

                if user_question:
                    output_placeholder = st.empty()  # Prepare a placeholder for the output

                    for chunk in rag_chain_with_source.stream(user_question):
                        for key, value in chunk.items():
                            if key not in output:
                                output[key] = value
                            else:
                                output[key] += value

                        if output.get('answer'):
                            with output_placeholder.container():
                                st.markdown(f"**👾:** {output['answer']}")
                            st.session_state.last_response = output['answer']

                    if output:
                        new_data = pd.DataFrame([{"timestamp": datetime.now(), "question": user_question, "answer": output["answer"]}])
                        st.session_state.user_history = pd.concat([st.session_state.user_history, new_data], ignore_index=True)
                        ChatManager.save_user_history(username, new_data)
                        st.experimental_rerun()

        # Main Chat Area (Below the input)
        chat_container = st.container()

        # Display chat history in reverse order
        for _, row in st.session_state.user_history.iloc[::-1].iterrows():
            with chat_container:
                st.markdown(f"**😊:** {row['question']}")
                st.markdown(f"**👾:** {row['answer']}")
                st.divider()  # Add a divider between messages

        # Sidebar: View History
        if "sidebar_search_history_query" not in st.session_state:
            st.session_state.sidebar_search_history_query = ""

        with st.sidebar.expander("Chat History", expanded=False):
            sidebar_search_query = st.text_input(
                "It matters !!!", 
                value=st.session_state.sidebar_search_history_query, 
                placeholder="Start Digging into it...", 
                key="sidebar_search_history_query"
            )

            filtered_history = st.session_state.user_history[
                st.session_state.user_history.apply(lambda row: st.session_state.sidebar_search_history_query.lower() in row.to_string().lower(), axis=1)
            ] if st.session_state.sidebar_search_history_query else st.session_state.user_history
            if st.button("Clear Chat", key="clear_chat_button_sidebar"):
                backup_chat_history(username)  # Backup chat history before clearing
                ChatManager.clear_user_history(username)
                st.session_state.user_history = pd.DataFrame(columns=["timestamp", "question", "answer"])
                st.session_state.user_question = ""
                st.session_state.last_question = ""
                st.experimental_rerun()  # Refresh the page to sync changes
            for _, row in filtered_history.iterrows():
                st.write(f"**😊:** {row['question']}")
                st.write(f"**👾:** {row['answer']}")

    elif page == "Feedback":        
        feedback_page(username)
        
    elif page == "About":
        image_url = "/home/harshk/LLM/modular/c.png"
        st.image(image_url, width=100, caption='')  
        st.write("ApnaGPT is a powerful tool for answering questions based on the context provided in a PDF document. Developed by Harsh Kumar at EduSwap Lab.")

    elif page == "Logout":
        del st.session_state.username
        st.experimental_rerun()


def feedback_page(username):
    st.markdown(
        '<div style="animation: pulse 2s infinite; text-align: center;">'
        '<h1 style="color: #3498DB;">We value your feedback!</h1>'
        '</div>',
        unsafe_allow_html=True
    )
    
    feedback = st.text_area("Please provide your feedback here:", height=200, placeholder="Write here...")

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
            feedback_filename = f"{username}_{timestamp}.txt"
            feedback_file_path = save_text_file(f"{emoji} {feedback}", FEEDBACK_FOLDER, feedback_filename)
            st.success(f"Feedback saved as {feedback_file_path}")
        else:
            st.error("Feedback cannot be empty!")

def landing_page():
    st.markdown("""
        <style>
            .main-header { /* Main header styling */
                font-size: 2.5rem;
                text-align: center;
                color: #2E86C1;
                margin-top: 20px;
                animation: fadeIn 2s ease-in-out;
            }
            .sub-header { /* Sub-header styling */
                font-size: 1.25rem;
                text-align: center;
                color: #34495E;
                margin: 10px 0;
                opacity: 0;
                animation: fadeIn 2s ease-in-out forwards;
                animation-delay: 1s;
            }
            .button-center { /* Button container styling */
                display: flex;
                justify-content: center;
                margin-top: 20px;
            }
            .button { /* Button styling */
                font-size: 1rem;
                padding: 10px 30px;
                color: white;
                background: linear-gradient(135deg, #2E86C1, #1B4F72);
                border: none;
                border-radius: 25px;
                cursor: pointer;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                transition: background-color 0.3s, transform 0.3s, box-shadow 0.3s;
                text-align: center;
                animation: pulse 2s infinite;
            }
            .button:hover { /* Button hover effect */
                background: linear-gradient(135deg, #1B4F72, #2E86C1);
                transform: scale(1.05);
                box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
            }
            @keyframes pulse { /* Animation for button pulsing effect */
                0% { transform: scale(1); }
                50% { transform: scale(1.05); }
                100% { transform: scale(1); }
            }
            @keyframes fadeIn { /* Animation for text fade-in */
                from { opacity: 0; }
                to { opacity: 1; }
            }
        </style>
        <div class="main-header">Welcome to ApnaGPT</div>
        <div class="sub-header" style="animation-delay: 0s;">
             👉 Please frame specific questions similar to the provided examples.<br>
             👉 If the response doesn't match your query, consider rephrasing the question.
        </div>
        
        <div class="button-center">
            <button class="button" onclick="window.location.href='/main'">Get Started</button>
        </div>
    """, unsafe_allow_html=True)