import streamlit as st
from authenticator import Authenticator
from ui import main_content, landing_page, feedback_page
from authenticator import Authenticator
from config import VECTORSTORE_PATH, MODEL_PATH

def main():
    st.set_page_config(page_title="ApnaGPT", page_icon="/home/harshk/LLM/modular/c.png")


    if "username" not in st.session_state:
        landing_page()
        st.sidebar.image("/home/harshk/LLM/modular/c.png", width=100)
        st.sidebar.title("Login / Signup")
        choice = st.sidebar.radio("Choose Action", ["Login", "Signup", "Forgot Password"])

        if choice == "Login":
            login_username = st.sidebar.text_input("Username")
            login_password = st.sidebar.text_input("Password", type="password")
            if st.sidebar.button("Login"):
                if Authenticator.authenticate_user(login_username, login_password):
                    st.session_state.username = login_username
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password")

        elif choice == "Signup":
            signup_username = st.sidebar.text_input("New Username")
            signup_password = st.sidebar.text_input("New Password", type="password")
            signup_phone = st.sidebar.text_input("Phone Number (10 digits)")
            if st.sidebar.button("Signup"):
                if len(signup_phone) == 10 and signup_phone.isdigit():
                    if Authenticator.register_user(signup_username, signup_password, signup_phone):
                        st.success("Signup successful! Please log in.")
                    else:
                        st.error("User already exists")
                else:
                    st.error("Invalid phone number. Please enter a 10-digit phone number.")

        elif choice == "Forgot Password":
            reset_phone = st.sidebar.text_input("Enter your phone number")
            if st.sidebar.button("Get Username"):
                username = Authenticator.get_username_by_phone(reset_phone)
                if username:
                    st.session_state.reset_username = username
                    st.success(f"Username found: {username}")
                else:
                    st.error("Phone number not found")

            if "reset_username" in st.session_state:
                reset_password = st.sidebar.text_input("Enter new password", type="password")
                if st.sidebar.button("Reset Password"):
                    if Authenticator.update_password(st.session_state.reset_username, reset_password):
                        st.success("Password reset successfully. Please log in.")
                        del st.session_state.reset_username
                    else:
                        st.error("Failed to reset password")

    else:
        main_content(st.session_state.username)

if __name__ == "__main__":
    main()
