import time
import markdown_conversion
import markdown_preprocessing
from langchain.schema import Document
import faiss
import numpy as np
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_core.messages import HumanMessage, AIMessage

# Display welcome message
if "output_path" in st.session_state:
    st.write("Hello, Welcome to the Question and Answer Page!")
else:
    st.write("Output Path not available.")

input_file_path = st.session_state.output_path

# Progress bar for loading
progress_bar = st.progress(0)
status_text = st.empty()

# Preprocess the markdown document
texts = markdown_preprocessing.preprocess_markdow_doc(input_file_path)

progress_bar.progress(20)
status_text.text("Processing the Document")

# Convert text into LangChain Documents
texts = [Document(page_content=text) for text in texts]

# Generate embeddings and create FAISS index
embeddings = OllamaEmbeddings(model="mistral:latest")
vector_store = FAISS.from_documents(texts, embeddings)

progress_bar.progress(100)
status_text.text("Document Processing Complete")

# Custom prompt template
custom_prompt_template = """You are a personalised Document Analyser. Use the following information to answer the user's question.
If you don't know the answer, just say that you don't know.

Context: {context}
Question: {question}

Only return the helpful answer below.
Helpful answer:
"""

prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

# Initialize LLM and retrieval-based QA
llm = ChatOllama(model="mistral:latest", base_url="http://localhost:11434", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=vector_store.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': prompt}
)

st.write("Chatbot is ready! Start asking questions below.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="I am a bot, how can I help you?")
    ]

# Chat Input
user_input = st.chat_input("Type your message here...")
if user_input and user_input.strip():
    response_dict = qa_chain.invoke(user_input)  # Returns {'result': ..., 'source_documents': ...}
    response = response_dict["result"]  # Extract only the answer

    # Append conversation to history
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    else:
        with st.chat_message("Human"):
            st.write(message.content)
