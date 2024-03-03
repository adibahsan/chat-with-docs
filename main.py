import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyMuPDFLoader
from transformers import AutoTokenizer

import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/oasst-sft-1-pythia-12b")


def load_doc(pdf_doc):
    loader = PyMuPDFLoader(pdf_doc.name)
    documents = loader.load()
    embedding = HuggingFaceEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text = text_splitter.split_documents(documents)
    db = Chroma.from_documents(text, embedding)
    llm = HuggingFaceHub(repo_id="OpenAssistant/oasst-sft-1-pythia-12b",
                         model_kwargs={"temperature": 0.2, "max_length": 4096})
    global chain
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
    return 'Document has successfully been loaded'


def extract_helpful_answer(answer_chunks):
    # Find the index of "Helpful Answer:"
    index = answer_chunks.find("Helpful Answer:")

    # If "Helpful Answer:" is not in the string, return an empty string
    if index == -1:
        return ""

    # Add the length of "Helpful Answer:" to the index to start from the end of that string
    index += len("Helpful Answer:")

    # Return the part of the string from the index to the end
    return answer_chunks[index:].strip()


# Usage
# answer_chunks = "..."  # Your string here
# helpful_answer = extract_helpful_answer(answer_chunks)
# print(helpful_answer)
def answer_query(query):
    # Tokenize the input text
    tokens = tokenizer.tokenize(query)

    # Split the tokens into chunks of 1024
    chunks = [tokens[i:i + 1024] for i in range(0, len(tokens), 1024)]

    # Process each chunk separately
    answers = []
    for chunk in chunks:
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        # Now you can pass `chunk_text` to your model

        answer_chunks = chain.run(chunk_text)
        modified_answer = extract_helpful_answer(answer_chunks)
        print(modified_answer)
        print("-----------")

        answers.append(modified_answer)

    return ' '.join(answers)


html = """
<div style="text-align:center; max width: 700px;">
    <h1>ChatPDF</h1>
    <p> Upload a PDF File, then click on Load PDF File <br>
    Once the document has been loaded you can begin chatting with the PDF =)
</div>"""
css = """container{max-width:700px; margin-left:auto; margin-right:auto,padding:20px}"""
with gr.Blocks(css=css, theme=gr.themes.Monochrome()) as demo:
    gr.HTML(html)
    with gr.Column():
        gr.Markdown('ChatPDF')
        pdf_doc = gr.File(label="Load a pdf or a docx file", file_types=['.pdf', '.docx'], type='filepath')
        with gr.Row():
            load_pdf = gr.Button('Load pdf file')
            status = gr.Textbox(label="Status", placeholder='', interactive=False)

        with gr.Row():
            input = gr.Textbox(label="type in your question")
            output = gr.Textbox(label="output")
        submit_query = gr.Button("submit")

        load_pdf.click(load_doc, inputs=pdf_doc, outputs=status)

        submit_query.click(answer_query, input, output)

demo.launch()
