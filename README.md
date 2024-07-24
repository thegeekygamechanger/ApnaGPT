# ApnaGPT

ApnaGPT is a Streamlit app that provides a powerful tool for answering questions based on the context provided in a PDF document. It utilizes various natural language processing (NLP) techniques and models to generate responses.

## Features

- **PDF Document Processing:** Add path of your PDF document, and the app extracts text from it and processes it for question answering.
- **Question Answering:** Users can ask questions based on the context provided in the PDF document, and the app generates answers using language models.
- **User Authentication:** Users can log in, sign up, or reset their passwords securely.
- **Chat History:** The app maintains a chat history for each user, allowing them to review past conversations.
- **Feedback Submission:** Users can provide feedback on the service, which is recorded and stored for analysis.

## Usage

1. Install the required dependencies:
langchain==0.3.0
pandas==1.3.3
streamlit==0.88.0

2. Run the Streamlit app:
streamlit run main.py

3. Access the app through the provided URL in your browser.

## Configuration

- **PDF_PATH:** Path to the PDF document for processing.
- **MODEL_PATH:** Path to the language model used for question answering.
- **CHAT_SESSION_FOLDER:** Folder for storing chat session data.
- **FEEDBACK_FOLDER:** Folder for storing user feedback.
- **VECTORSTORE_PATH:** Path to the vector store used for similarity search.
- **USER_DB:** File path for storing user authentication data.

## Requirements

- Python 3.8 or up
- Streamlit
- langchain (Python package)

## Author

[Harsh Kumar](https://github.com/thegeekygamechanger)

