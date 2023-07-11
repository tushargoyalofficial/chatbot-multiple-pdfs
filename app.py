import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

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
                    st.write(raw_text)

if __name__ == "__main__":
    main()
