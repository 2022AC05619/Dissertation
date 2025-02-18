import time
import markdown_conversion
import markdown_preprocessing
from langchain.schema import Document
import faiss
import numpy as np
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
import streamlit as st


if "output_path" in st.session_state:
    st.write(f"Hello, Welcome to Question and Answer Page")
else:
    st.write("Output Path not available.")

input_file_path = st.session_state.output_path


# input_file_path = output_path
input_file_path = input_file_path
texts = markdown_preprocessing.preprocess_markdow_doc(input_file_path)

texts = [Document(page_content=texts[i]) for i in range(len(texts))] 

embeddings = OllamaEmbeddings(
            model="mistral:latest"
            # embed_dimension=1024  # Mistral's embedding dimension
        )

vector_store = FAISS.from_documents(texts, embeddings)

custom_prompt_template = """You are a personalised Document Analyser .Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

prompt = PromptTemplate(template=custom_prompt_template,
                        input_variables=['context', 'question'])

llm2 = ChatOllama(model="mistral:latest",base_url="http://localhost:11434", temperature=0)

qa_chain = RetrievalQA.from_chain_type(llm=llm2,
                                   chain_type='stuff',
                                   retriever=vector_store.as_retriever(search_kwargs={'k': 3}),
                                   return_source_documents=True,
                                   chain_type_kwargs={'prompt': prompt}
                                      )


st.write("Chatbot is ready! Type 'exit' to quit.")
while True:
    query = st.text_input("You:")
    if query.lower() == 'exit':
        break
    if st.button("Submit") and query:
        response = qa_chain.invoke(query)
        st.write("**Bot:**", response)