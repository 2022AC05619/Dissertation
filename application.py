import streamlit as st
import markdown_conversion
import os

os.environ["OMP_NUM_THREADS"] = "3"

st.title("PDF Q&A Chatbot")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
output_path = ""

progress_bar = st.progress(0)
if uploaded_file is not None:
    progress_bar.progress(20)
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())
        output_path = markdown_conversion.conver_to_markdown("F:\Bits\Sem 4 - Dissertation","uploaded")
        #output_path = "F:\Bits\Sem 4 - Dissertation\data\lambda-dg-300.md"
        progress_bar.progress(100)

        if st.button("Summary"):
            st.session_state.output_path = output_path
            st.switch_page("pages/Summary.py")
        if st.button("Q&A"):
            st.session_state.output_path = output_path
            st.switch_page("pages/Question_answer.py") 


st.write("**Bot:**", output_path)
