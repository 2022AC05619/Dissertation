import time
import markdown_conversion
import markdown_preprocessing
import os
import map_reduce
# Load model directly
from langchain_ollama import ChatOllama
from langchain import PromptTemplate
from langchain_core.prompts import PromptTemplate
# Summarizer we'll use for Map Reduce
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from langchain.chains import LLMChain
import torch
import gc
import streamlit as st

from itertools import islice

def batch_process(iterable, batch_size):
    it = iter(iterable)
    while batch := list(islice(it, batch_size)):
        yield batch


### Pre Processing


# Set OMP_NUM_THREADS to 3
os.environ["OMP_NUM_THREADS"] = "3"
model_name = "hf.co/saishshinde15/TethysAI_Research:Q8_0"

progress_bar = st.progress(0)
status_text = st.empty()
# output_path = markdown_conversion.conver_to_markdown("F:\Bits\Sem 4 - Dissertation\data","lambda-dg-300")

## Clear cache
torch.cuda.empty_cache()

if "output_path" in st.session_state:
    st.write(f"Hello, {st.session_state.output_path}! Welcome to Summary Page.")
else:
    st.write("Output Path not available.")

input_file_path = st.session_state.output_path
# input_file_path = "F:\Bits\Sem 4 - Dissertation\data\lambda-dg-300.md"
texts = markdown_preprocessing.preprocess_markdow_doc(input_file_path)
progress_bar.progress(15)
status_text.text("Pre-processing the markdown file")

new_texts = markdown_preprocessing.text_chunking(texts,3000)

progress_bar.progress(20)
status_text.text("Text chunking")




### MAP Reduce 1

summary_list_1 = map_reduce.Map_Reduce_summary(model_name,new_texts[:int(len(new_texts)/2)])

progress_bar.progress(35)
status_text.text("Map Reduce 1")

summary_list_2 = map_reduce.Map_Reduce_summary(model_name,new_texts[int(len(new_texts)/2):])

progress_bar.progress(45)
status_text.text("Map Reduce 1 completed")

# Clear cache
torch.cuda.empty_cache()






### MAP Reduce 2

llm2 = ChatOllama(model="mistral:latest",base_url="http://localhost:11434", temperature=0)

summary_list = summary_list_1+summary_list_2

summaris_doc = [Document(page_content=summary_list[i]) for i in range(len(summary_list))]

progress_bar.progress(50)
status_text.text("Map Reduce 2 starting")
combine_prompt = """
You are an expert at writing Summries.
You will be given a series of summaries from set of Documents. The summaries will be enclosed in triple backticks (```)
Your goal is to give a verbose summary of what happened in the Text.
```{text}```
VERBOSE SUMMARY:
"""
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

reduce_chain = LLMChain(llm=llm2,
                             # chain_type="stuff",
                             prompt=combine_prompt_template
                                )

reduced_summaries = []



for batch in batch_process(summaris_doc, 10):
    reduced_summaries.append(reduce_chain.run(batch))

progress_bar.progress(75)
status_text.text("Map Reduce 2 completed")







### MAP Reduce 3



model_name = "hf.co/saishshinde15/TethysAI_Research:Q8_0"
final_summary = map_reduce.Map_Reduce_summary(model_name,reduced_summaries)

Final_summaries = ""
for i in range(len(final_summary)):
    Final_summaries += " Summary "+str(i)+ ": \n"+final_summary[i]+"\n\n"

progress_bar.progress(90)
status_text.text("Map Reduce 3 completed")


Final_summaries = Final_summaries.replace("Summary","")
st.write("**Summary: **",Final_summaries)
# ### Final Formating

# combine_prompt2 = """
# You are an expert in formatting Summaries. 
# Each Summaries start with "Summaries : ", and the mapped summaries are given below:
# {text}
# Format the summaries to 4 to 5 paragraphs.
# Helpful Answer:
# """
# map_prompt_template = PromptTemplate(template=combine_prompt2, input_variables=["text"])
# reduce_chain_2 = LLMChain(llm=llm2,
#                             # chain_type="stuff",
#                              prompt=map_prompt_template,
#                              # output_key='Proper_summary',
#                               # verbose=True
#                                    )

# output_eg_2 = reduce_chain_2.run([Final_summaries])

# progress_bar.progress(100)
# status_text.text("Processing completed")
# # print(output_eg_2)
# st.write("**Summary:**", output_eg_2)
