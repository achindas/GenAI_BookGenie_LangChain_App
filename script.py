# Step 1: Import All the Required Libraries

# We are creating a Webapp with Streamlit
import streamlit as st

# Replicate is an online cloud platform that allows us to host models and access the models through API
#Llama 2 models with 7B, 13 B and with 70B parameters are hosted on Replicated and we will access these models through API

import replicate
import os

# Import langchain and pinecone libraries

# from langchain_community.document_loaders import PyPDFLoader, OnlinePDFLoader
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain
#from langchain.llms import HuggingFaceHub
from langchain_community.llms import HuggingFaceHub
import time

# Setup the Environment
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ['PINECONE_API_KEY'] = 'pcsk_3xCWdh_2NrnyFTFipGpPEkt4YB1JV3E95VuJ2MLUYF39dn6RQyCFdYNQaJiMKUuPhwStAa'

# Next Step: Create a Streamlit Application on Browser

st.set_page_config(page_title="🦙💬 Llama 2 Chatbot with Streamlit")

# Define a function to extract text from PDF
def extract_text_from_pdf(pdf):
    pdf_reader = PdfReader(pdf) 
    #return ''.join(page.extract_text() for page in pdf_reader.pages)
    return [page.extract_text() for page in pdf_reader.pages]


# Define a function to process the PDF book
def process_text_in_pdf(data):

    # Downlaod the Embeddings
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    index_name = "bookgenie" # put in the name of pinecone index here

    # If vector_action is None, use existing vector
    from langchain_community.vectorstores import Pinecone
    if st.session_state.vector_action == "None": 
        docsearch = Pinecone.from_existing_index(index_name, embeddings)
        # Perform a similarity search with a generic query
        query = "What is the main idea?"  # Generic query
        results = docsearch.similarity_search(query, k=5)  # Retrieve top 5 most similar vectors

        # Check results
        if len(results) > 0:
            print("Embedding vectors retrieved successfully!")
            return docsearch
        else:
            print("No embedding vectors retrieved. The query result is empty.")
            return None
    
    # If vector_action is Delete or Replace, then delete the existing vector
    if st.session_state.vector_action in ["Delete", "Replace"]:
        
        from pinecone import Pinecone

        pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
        index = pc.Index(name=index_name)
        try:
            index.delete(delete_all=True, namespace='')
            if st.session_state.vector_action == "Delete": return None
        except Exception as e:
            print(f"Error deleting index '{index_name}': {e}")
            if st.session_state.vector_action == "Delete": return None

    # Below steps are required if vector_action is Add or Replace
    # Split the book text into Chunks
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs=text_splitter.create_documents(data)
    print("\nSplit Doc Length: ", len(docs))

    # Create Embeddings for Each of the Text Chunk
    from langchain_community.vectorstores import Pinecone
    docsearch = Pinecone.from_texts(
        [t.page_content for t in docs],
        embeddings,
        index_name=index_name
    )

    return docsearch

# Define a function to generate file upload part of sidebar
# @st.cache_resource
def create_book_vector(uploaded_file):

    if uploaded_file is not None:
        
        # Display the progress bar
        progress_bar = st.progress(0)
        status_message = st.empty()
        status_message.text("Book uploaded. Processing the PDF file...")

        # Simulate processing of the PDF with a loop
        for progress in range(100):
            time.sleep(0.05)  # Simulating processing delay
            progress_bar.progress(progress + 1)

        book_data = extract_text_from_pdf(uploaded_file)
        # print("Data:", book_data)
        print("Doc Length: ", len(book_data))

        # Obtain the embeddings for the book
        book_search = process_text_in_pdf(book_data)

        # Replace progress bar with the success message once the file is processed
        progress_bar.empty()  # Clear the progress bar
        status_message.empty()  # Clear the status message
        if book_search is None:
            st.success("Book is processed! Book vector is not available" )
        else:
            st.success("Book is processed! Book vector is available" )

        return book_search

#Create a Side bar

# Initialize session state for file uploader
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = None

with st.sidebar:
    st.title("🦙💬 BookGenie Chatbot")
 
    st.header("Book Settings")

    # Add flags regarding book vector action
    st.radio(
        "Set Vector Action for Book 👇",
        ["Add", "Replace", "Delete", "None"],
        key="vector_action",
        horizontal=True,
        index=3
    )

    # print("Radio Selected: ", st.session_state.vector_action)

    # UpLoad & process the PDF Book for conversation
    uploaded_file = st.file_uploader("Choose your book in PDF format", type="pdf")
    print("File uploade state: ", st.session_state.file_uploaded)
    if uploaded_file is not None:
        if st.session_state.file_uploaded != uploaded_file.name:
            # Update the session state after the check
            st.session_state.file_uploaded = uploaded_file.name
            book_vector = create_book_vector(uploaded_file)
            print("Book Vector: ", book_vector)
            # Store the book vector in Session for use across the app
            st.session_state.book_vector = book_vector    
    
    # Add a checkbox for enabling chatting unrelated to the uploaded book
    st.session_state.enable_general_chat = st.checkbox("Enable general chat (unrelated to book)", value=False)

    st.header("Model Settings")

    add_replicate_api=st.text_input('Enter the Replicate API token', type='password')
    if not (add_replicate_api.startswith('r8_') and len(add_replicate_api)==40):
        st.warning('Please enter your credentials', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')

    st.subheader("Models and Parameters")

    select_model=st.selectbox("Choose a Llama 2 Model", ['Llama 2 7b', 'Llama 2 13b', 'Llama 2 70b'], key='select_model')
    if select_model=='Llama 2 7b':
        llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
    elif select_model=='Llama 2 13b':
        llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
    else:
        llm = 'replicate/llama70b-v2-chat:e951f18578850b652510200860fc4ea62b3b16fac280f83ff32282f87bbd2e48'

    temperature=st.slider('temperature', min_value=0.01, max_value=5.0, value=0.5, step=0.01)
    top_p=st.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length=st.slider('max_length', min_value=64, max_value=4096, value=512, step=8)

    st.markdown('Please check my project portfolio [link](https://achindas.github.io/portfolio/)')

os.environ['REPLICATE_API_TOKEN']=add_replicate_api

#Store the LLM Generated Reponese

if "messages" not in st.session_state.keys():
    st.session_state.messages=[{"role": "assistant", "content":"How may I assist you today?"}]

# Diplay the chat messages

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# Clear the Chat Messages
def clear_chat_history():
    st.session_state.messages=[{"role":"assistant", "content": "How may I assist you today"}]

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


# Create a Function to generate the Llama 2 Response
def generate_llama2_response(prompt_input):
    default_system_prompt="You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for data in st.session_state.messages:
        print("Data:", data)
        if data["role"]=="user":
            default_system_prompt+="User: " + data["content"] + "\n\n"
        else:
            default_system_prompt+="Assistant" + data["content"] + "\n\n"
    output=replicate.run(llm, input={"prompt": f"{default_system_prompt} {prompt_input} Assistant: ",
                                     "temperature": temperature, "top_p":top_p, "max_length": max_length, "repititon_penalty":1})

    return output


# Create a Function to generate the LLM Response
def generate_llm_response(prompt_input):
    default_system_prompt="You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for data in st.session_state.messages:
        print("Data:", data)
        if data["role"]=="user":
            default_system_prompt+="User: " + data["content"] + "\n\n"
        else:
            default_system_prompt+="Assistant" + data["content"] + "\n\n"

    llm=HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":temperature, "max_length":max_length})
    chain=load_qa_chain(llm, chain_type="stuff")
    docs=st.session_state.book_vector.similarity_search(prompt_input)
    print("Searched Docs: ", docs)
    output = chain.run(input_documents=docs, question=prompt_input)

    return output


#User -Provided Prompt

chat_input_flag = False
# Determine if chat input should be enabled
if st.session_state.enable_general_chat is True:
    if add_replicate_api:
        chat_input_flag = True
else:
    if st.session_state.file_uploaded is not None:
        if st.session_state.book_vector is not None:
            chat_input_flag = True

if prompt := st.chat_input(
    "Ask a question here", 
    disabled=not chat_input_flag
    ):
    st.session_state.messages.append({"role": "user", "content":prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a New Response if the last message is not from the asssistant
# Call the right function based on status of checkbox
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if st.session_state.get("enable_general_chat", False):
                # Call generate_llama2_response for general chat
                response = generate_llama2_response(prompt)
            else:
                # Call generate_llm_response for book-related chat
                response = generate_llm_response(prompt)
            placeholder=st.empty()
            full_response=''
            for item in response:
                full_response+=item
                # placeholder.markdown(full_response)
            placeholder.markdown(full_response)

    message= {"role":"assistant", "content":full_response}
    st.session_state.messages.append(message)





