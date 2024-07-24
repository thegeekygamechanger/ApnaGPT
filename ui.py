import streamlit as st
import pandas as pd
from datetime import datetime
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
from config import CHAT_SESSION_FOLDER, FEEDBACK_FOLDER, VECTORSTORE_PATH, MODEL_PATH, ENCODER_MODEL_PATH, LOGO_PATH

# Define ChatManager class
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
    def clear_user_main_chat():
        st.session_state.main_chat = pd.DataFrame(columns=["timestamp", "question", "answer"])

    @staticmethod
    def clear_user_history(username):
        filepath = os.path.join(CHAT_SESSION_FOLDER, f"{username}_history.csv")
        if os.path.exists(filepath):
            os.remove(filepath)
        st.session_state.user_history = pd.DataFrame(columns=["timestamp", "question", "answer"])
        st.session_state.main_chat = pd.DataFrame(columns=["timestamp", "question", "answer"])
    
    @staticmethod
    def clear_all_chat(username):
        filepath = os.path.join(CHAT_SESSION_FOLDER, f"{username}_history.csv")
        if os.path.exists(filepath):
            os.remove(filepath)
        # Clear in session state
        st.session_state.user_history = pd.DataFrame(columns=["timestamp", "question", "answer"])
        st.session_state.main_chat = pd.DataFrame(columns=["timestamp", "question", "answer"])


# Function to setup RAG chain
def setup_rag_chain(_retriever):
    prompt_template = """
    <s>[INST] You are a C-Bot Assitant developed by Harsh Kumar at C-DAC Pune.

    {context}
    You are a respectful and honest C-BOT assistant. Answer the user's questions using only the context provided to you. Answer coding-related questions with code and explanation. You should not start giving code to yourself, if someone ask to write any code, avoid giving code. Do not start the response with salutations, answer directly. Do not start generating random code from context provided to you. If any user ask you "who" starting question, answer them Harsh Kumar Developed you.
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

# Function to backup chat history
def backup_chat_history(username):
    backup_folder = os.path.join(FEEDBACK_FOLDER, "backup")
    os.makedirs(backup_folder, exist_ok=True)
    backup_filename = os.path.join(backup_folder, f"{username}_chat_backup_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv")
    st.session_state.user_history.to_csv(backup_filename, index=False)

# Function for main content
# Function for main content
def main_content(username):
    st.sidebar.image(LOGO_PATH, width=100)  # Replace with your image URL and width
    st.sidebar.markdown(f'<h1 style="color: Navy;">Welcome, {username}</h1>', unsafe_allow_html=True)
    page = st.sidebar.radio("", ["Home", "Feedback", "About", "Logout"])

    # Initialize user history and main chat data if not already initialized
    if "user_history" not in st.session_state or st.session_state.get("current_username") != username:
        st.session_state.user_history = ChatManager.load_user_history(username)
        st.session_state.current_username = username

    if "main_chat" not in st.session_state:
        st.session_state.main_chat = pd.DataFrame(columns=["timestamp", "question", "answer"])

    # Handle different pages in the sidebar
    if page == "Home":
        # Initialize retriever and RAG chain if not already initialized
        if "retriever" not in st.session_state:
            st.session_state.retriever = initialize_retriever(VECTORSTORE_PATH, MODEL_PATH, ENCODER_MODEL_PATH)
        _retriever = st.session_state.retriever

        if "rag_chain_with_source" not in st.session_state:
            st.session_state.rag_chain_with_source = setup_rag_chain(_retriever)
        rag_chain_with_source = st.session_state.rag_chain_with_source

        # User input section
        with st.form(key='user_input_form', clear_on_submit=True):
            user_question = st.text_input("Ask your question:", key='user_question_input', placeholder="Write Prompt")
            submit_button_clicked = st.form_submit_button("Submit")

            if submit_button_clicked:
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
                            with output_placeholder.container():
                                st.markdown(
                                    f'<div style="border: 2px solid blue; padding: 10px; border-radius: 5px; background-color: #e0f7fa;">'
                                    f'<strong>😊 Question:</strong> {user_question}<br>'
                                    f'<strong>👾 Answer:</strong> {output["answer"]}'
                                    f'</div>', unsafe_allow_html=True
                                )
                            st.session_state.last_response = output['answer']

                    if output:
                        new_data = pd.DataFrame([{"timestamp": datetime.now(), "question": user_question, "answer": output["answer"]}])
                        st.session_state.user_history = pd.concat([st.session_state.user_history, new_data], ignore_index=True)
                        ChatManager.save_user_history(username, new_data)
                        # Update the main chat in the session state
                        st.session_state.main_chat = pd.concat([st.session_state.main_chat, new_data], ignore_index=True)

        # Main chat area
        chat_container = st.container()
        for _, row in st.session_state.main_chat.iterrows():
            with chat_container:
                st.write(f"**[{row['timestamp']}] 🙋 Question:** {row['question']}")
                st.markdown(f"**👾 Answer:** {row['answer']}")
                st.divider()

        # Sidebar: View history
        with st.sidebar.expander("Chat History", expanded=False):
            sidebar_search_query = st.text_input(
                "Search history",
                value=st.session_state.get("sidebar_search_history_query", ""),
                placeholder="Start digging into it..."
            )

            filtered_history = st.session_state.user_history[
                st.session_state.user_history.apply(lambda row: sidebar_search_query.lower() in row.to_string().lower(), axis=1)
            ] if sidebar_search_query else st.session_state.user_history

            if st.button("Clear All Chat", key="clear_all_chat_button_sidebar"):
                ChatManager.clear_all_chat(username)
                # Clear the chats immediately without reloading the page
                st.session_state.main_chat = pd.DataFrame(columns=["timestamp", "question", "answer"])
                st.session_state.user_history = pd.DataFrame(columns=["timestamp", "question", "answer"])
                st.experimental_rerun()

            for _, row in filtered_history.iterrows():
                st.write(f"**[{row['timestamp']}]**")
                st.write(f"**Question:** {row['question']}")
                st.write(f"**👾 Answer:** {row['answer']}")

    elif page == "Feedback":
        feedback_page(username)

    elif page == "About":
        st.image(LOGO_PATH, width=100, caption='')
        st.write("ApnaGPT is a powerful tool for answering questions based on the context provided, it is developed by Harsh Kumar  .")

    elif page == "Logout":
        del st.session_state.username
        st.session_state.main_chat = pd.DataFrame(columns=["timestamp", "question", "answer"])
        st.session_state.user_question = ""
        st.session_state.last_response = ""
        # Clear main chat on logout
        st.experimental_rerun()


# Function for feedback page
def feedback_page(username):
    st.markdown(
        '<div style="animation: pulse 2s infinite; text-align: center;">'
        '<h3 style="color: #3498DB;">We value your feedback!</h3>'
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

    submit_feedback = st.button("Submit Feedback")

    if submit_feedback and feedback.strip():
        save_text_file(os.path.join(FEEDBACK_FOLDER, f"{username}_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"), feedback)
        st.success("Thank you for your feedback!")

# Function for landing page
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
        <div class="main-header">Welcome to CDAC Mitra</div>
        <div class="button-center">
            <button class="button" onclick="window.location.href='/main'"><p>👉 Create User login<br><br>
                👉 Please frame specific questions similar to the provided examples<br><br>
                👉 If the response doesn't match your query, consider rephrasing the questions</button><br><br>
                ✨It is Gen AI Powered Query Enhancement System✨<br><br>
                ⚠️ Disclaimer: This bot does not have memory elements, so it will not be able to connect with previous questions !<br><br>📈 All the Best, waiting to hear from YOU ALL 🫵</p>
               </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    landing_page()
