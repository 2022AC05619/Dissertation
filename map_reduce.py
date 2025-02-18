# Load model directly
from langchain_ollama import ChatOllama
from langchain import PromptTemplate
from langchain_core.prompts import PromptTemplate
# Summarizer we'll use for Map Reduce
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document



def Map_Reduce_summary(model_name,Document_list):

    # llm = ChatOllama(model="hf.co/saishshinde15/TethysAI_Research:Q8_0",base_url="http://localhost:11434", temperature=0)
    llm = ChatOllama(model=model_name,base_url="http://localhost:11434", temperature=0)



    map_prompt = """
    Your are a helpful chatbot and an expert in extracting the main themes from the given document.
    You have been provided a set of documents below
    ```{text}```
    based on the documents, please identify the main themes and give a paragraph for each.
    Helpful Answer:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    map_chain = load_summarize_chain(llm=llm,
                                chain_type="map_reduce",
                                # prompt=map_prompt_template
                                    )
    
    selected_docs = [Document(page_content=Document_list[i]) for i in range(len(Document_list))] 

    # Make an empty list to hold your summaries
    summary_list = []

    # Loop through a range of the lenght of your selected docs
    for i, doc in enumerate(selected_docs):

        # print(doc.page_content)
        # Go get a summary of the chunk
        chunk_summary = map_chain.run([doc])
        
        # Append that summary to your list
        summary_list.append(chunk_summary)
        
        # print (f"Summary #{i} (chunk #{selected_indices[i]}) - Preview: {chunk_summary} \n")

    return summary_list