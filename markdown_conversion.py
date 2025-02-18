import time
# Using your existing PDF conversion code
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered    


def conver_to_markdown(pdf_path,file_name):
    


    # Convert PDF to text
    print("converter")
    PdfConverter.use_llm = True
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )

    input_path = pdf_path+"\\"+file_name+".pdf"
    print("Render")
    rendered = converter(input_path)
    print("Main")
    strt_time = time.time()
    text, _, images = text_from_rendered(rendered)
    end_time = time.time()
    time_taken = end_time - strt_time

    output_file_path = pdf_path+"\\"+file_name+".md"
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(text)

    print(f"Text successfully extracted and saved to {output_file_path}")
    print(f"Time taken: {time_taken:.2f} seconds")

    return output_file_path