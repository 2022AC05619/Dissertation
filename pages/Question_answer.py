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
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from rank_bm25 import BM25Okapi
from llama_index.core import VectorStoreIndex
import os

def combined_context(vector_store,keyword_retriever, query, k=10):
    retriever_vectordb = vector_store.as_retriever(search_kwargs={"k": k})
    keyword_retriever.k = k
    ensemble_retriever = EnsembleRetriever(retrievers=[retriever_vectordb,keyword_retriever],
                                       weights=[0.5, 0.5])
    docs_rel=ensemble_retriever.get_relevant_documents(query)

    context_data = ""
    for i in range(len(docs_rel)):
        context_data += docs_rel[i].page_content + "\n\n"
    
    return context_data


# Create chatbot function with hybrid retrieval
def chatbot_response(vector_store,keyword_retriever , llm, query,):
    context = combined_context(vector_store,keyword_retriever, query, k=10)
    prompt_text = prompt.format(context=context, question=query)
    # print(prompt_text)
    response = llm.invoke(prompt_text)
    return response




def FAISS_file_exists(file_path):
    for filename in os.listdir(file_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):  # Ensure it's a file, not a directory
            print(file_path)


def FAISS_file_exists(file_path):
    file_ind = 0
    for filename in os.listdir(file_path):
        # print(filename)
        if filename == 'index.faiss':
            file_ind = 1
            break
        else:
            file_ind = 0
    return file_ind



file_location = "F:\Bits\Sem 4 - Dissertation"
Faiss_ind = FAISS_file_exists(file_location)
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
new_texts = markdown_preprocessing.text_chunking(texts,1000)

progress_bar.progress(30)
status_text.text("Processing the Document")

# Convert text into LangChain Documents
texts_new = [Document(page_content=new_texts[i]) for i in range(len(new_texts))]

# Generate embeddings and create FAISS index
if Faiss_ind == 1:
    print("File found")
    embeddings = OllamaEmbeddings(model="mistral:latest")
    vector_store = FAISS.load_local(file_location,embeddings,allow_dangerous_deserialization=True)
    keyword_retriever = BM25Retriever.from_documents(texts_new)

else:
    print("File not found")
    embeddings = OllamaEmbeddings(model="mistral:latest")
    vector_store = FAISS.from_documents(texts_new, embeddings)
    keyword_retriever = BM25Retriever.from_documents(texts_new)
    vector_store.save_local(file_location)

progress_bar.progress(100)
status_text.text("Document Processing Complete")

# Custom prompt template
custom_prompt_template = """You are a personalised Document Analyser. Use the following information to answer the user's question.
If you don't know the answer, just say that you don't know.

Context: 

{context}

Question: 

{question}

Only return the helpful answer below.
Helpful answer:
"""

prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

# Initialize LLM and retrieval-based QA
llm = ChatOllama(model="mistral:latest", base_url="http://localhost:11434", temperature=0)


st.write("Chatbot is ready! Start asking questions below.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="I am a bot, how can I help you?")
    ]

# Chat Input
user_input = st.chat_input("Type your message here...")
if user_input and user_input.strip():
    response_dict = chatbot_response(vector_store,keyword_retriever, llm, user_input)  # Returns {'result': ..., 'source_documents': ...}
    response = response_dict.content  # Extract only the answer

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
