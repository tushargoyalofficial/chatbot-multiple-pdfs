import streamlit as st

def main():
    st.set_page_config(page_title='Multiple PDFs Chatbot', page_icon=':books:')
    st.header('Multiple PDFs chatbot')
    st.text_input('Ask a question about your document(s)')

    with st.sidebar:
        st.subheader('Your documents')
        st.file_uploader(
            'Uplaod your PDFs & click on process',
            accept_multiple_files=True
        )
        st.button("Process")

if __name__ == "__main__":
    main()
