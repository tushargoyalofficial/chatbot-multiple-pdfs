import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000, # length of char
        chunk_overlap=200, # first chunk last 200 char === second chunk first 200 char
        length_function=len # default python len function
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

# the below method is not free, it's chargable
def using_openAI(text_chunks):
    embeddings = OpenAIEmbeddings();
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# this one, again we are going to create embeddings but for free, using our machine
# using instructor embeddings

def get_vectorstore(text_chunks):
    vector_openai = using_openAI(text_chunks)
    return vector_openai

def main():
    # loading variables inside .env file
    load_dotenv()

    st.set_page_config(page_title='Multiple PDFs Chatbot', page_icon=':books:')
    st.header('Multiple PDFs chatbot')
    st.text_input('Ask a question about your document(s)')

    with st.sidebar:
        st.subheader('Your documents')
        pdf_docs = st.file_uploader(
            'Uplaod your PDFs & click on process',
            type=['pdf'],
            accept_multiple_files=True
        )
        if st.button("Process"):
            if pdf_docs is not None:
                with st.spinner("Processing..."):
                    # get text out of pdf
                    raw_text = get_pdf_text(pdf_docs)
                    # get raw text chunks
                    text_chunks = get_text_chunks(raw_text)
                    # create vector store
                    vector_store = get_vectorstore(text_chunks)

if __name__ == "__main__":
    main()
