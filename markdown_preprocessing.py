import time
from langchain_text_splitters import MarkdownHeaderTextSplitter

def preprocess_markdow_doc(input_path):
    input_file_path = input_path
    with open(input_file_path, "r", encoding="utf-8") as file:
        text = file.read()

    print(f"Text successfully loaded to text")

    headers_to_split_on = [
        ("#", "Header 1")   
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
    md_header_splits = markdown_splitter.split_text(text)


    i = 0
    for i,split in enumerate(md_header_splits):
        value = list(split.metadata.values())    
        if len(value) == 0:
            continue
        header_text = value[0]
        toc_keywords = ['table of contents', 'contents', 'toc']
        if header_text.lower() in toc_keywords:
            print("found header")
            break

    del md_header_splits[i]
    texts = []

    for i,split in enumerate(md_header_splits):
        texts.append(split.page_content)

    return texts

def text_chunking(texts,chunk_limit):
    new_texts = []
    limit = chunk_limit
    result = ""
    for i in range(len(texts)):
        if i != 0:
            if len(texts[i]) >= limit:
                if len(result) != 0:
                    new_texts.append(result)
                new_texts.append(texts[i])
                result = ""
            else:
                result = result + '\\n' + texts[i]
                if len(result) >= limit:
                    if len(result) != 0:
                        new_texts.append(result)
                    result = ""
        else:
            if len(texts[i]) >= limit:
                new_texts.append(texts[i])
            else:
                result = texts[i]
    return new_texts