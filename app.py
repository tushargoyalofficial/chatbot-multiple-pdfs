import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
# from langchain.chat_models import ChatOpenAI

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
def using_open_ai(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# this one, again we are going to create embeddings but for free, using our machine
# using instructor embeddings
def using_huggingface_instruct(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# above one is too heavy and require very high configuration machine. Let's go with something easy 
# we will be using huggingface sentence transformer
def using_huggingface_sentencetransformers(text_chunks):
    model = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
    embeddings = SentenceTransformerEmbeddings(model_name=model)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# below we can create vector_db using 3 methods, depending on some pre-requisits
def get_vectorstore(text_chunks):
    vector_db = using_huggingface_sentencetransformers(text_chunks)
    return vector_db

def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def get_question_response(prompt):
    response = st.session_state.conversation({'question': prompt})
    st.write(response)

def main():
    # loading variables inside .env file
    load_dotenv()

    st.set_page_config(page_title='Multiple PDFs Chatbot', page_icon=':books:')
    st.header('Multiple PDFs chatbot')
    prompt = st.text_input('Ask a question about your document(s)')
    if prompt:
        get_question_response(prompt)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

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
                    vectorstore = get_vectorstore(text_chunks)
                    # create conversation chain (st.session_state to persist the variable as stramlit reloads all on button press)
                    # also it make the variable access outside of a function
                    st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == "__main__":
    main()
