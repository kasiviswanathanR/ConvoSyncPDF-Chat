import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Replicate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import Docx2txtLoader
import os
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="ConvoSyncPDF Chat",
    page_icon=":books:",
    layout="wide",  # Set layout to wide for a better UI
)

custom_styles = """
<style>
    /* Add your custom CSS styles here */
    body {
        font-family: 'Arial', sans-serif;
    }

    .streamlit-title {
        color: #1f68c1 !important;
    }

    /* Customize chat messages */
    .message-bubble {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
        max-width: 80%;
    }

    .message-bubble.user {
        background-color: #d6e9f5;
    }
</style>
"""
# Function to initialize session state
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["How can I help you ?"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey!"]

# Function for chat conversation
def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

# Function to display chat history
def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Documents", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

# Function to create conversational chain
def create_conversational_chain(vector_store):
    load_dotenv()
    llm = Replicate(
        streaming=True,
        model="replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781",
        callbacks=[StreamingStdOutCallbackHandler()],
        input={"temperature": 0.01, "max_length": 500, "top_p": 1})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)
    return chain

# Main function
def main():
    load_dotenv()
    # Initialize session state
    initialize_session_state()
    st.title("ConvoSyncPDF Chat:books:")

    # Sidebar configuration
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)

    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == ".docx" or file_extension == ".doc":
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
        text_chunks = text_splitter.split_documents(text)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                           model_kwargs={'device': 'cpu'})

        # Create vector store
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

        # Create the chain object
        chain = create_conversational_chain(vector_store)

        # Display chat history
        display_chat_history(chain)

if __name__ == "__main__":
    main()
