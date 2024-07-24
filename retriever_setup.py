# retriever_setup.py
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import streamlit as st
from config import PDF_PATH, MODEL_PATH, VECTORSTORE_PATH

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

@st.cache_resource
def initialize_retriever(store_path, model_path):
    chunks = load_and_split_pdf(PDF_PATH)
    vectorstore = create_embeddings_from_chunks(chunks, model_path, store_path)
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 4})

def format_documents(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
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
