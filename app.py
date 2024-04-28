
import os
import xml.etree.ElementTree as ET
import streamlit as st
from pinecone import ServerlessSpec, PodSpec
from langchain_pinecone import PineconeVectorStore
from llama_index.readers.sec_filings import SECFilingsLoader
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, OPENAI_API_KEY
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import xml.etree.ElementTree as ET
import streamlit as st
#from dotenv import load_dotenv
from pinecone import ServerlessSpec, PodSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pinecone
from langchain_community.vectorstores import Pinecone

from logging import NullHandler

# Pinecone Setup
api_key = PINECONE_API_KEY
environment = PINECONE_ENVIRONMENT
use_serverless = os.environ.get("USE_SERVERLESS", "False").lower() == "true"
import os
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
# Configure Pinecone client
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
spec = ServerlessSpec(cloud='gcp-starter', region='us-east-1') if use_serverless else PodSpec(environment=environment)

# Define or choose your index name
index_name = 'fin-docs-chat'
openai_api_key = OPENAI_API_KEY

model_name = 'gpt-4'


index = pc.Index(index_name)

embeddings = OpenAIEmbeddings(
        openai_api_key=openai_api_key
    )

docsearch = Pinecone.from_existing_index(
    index_name=index_name, 
    embedding=embeddings
)


def semantic_search(query):
    
    llm = ChatOpenAI(model_name='gpt-4', api_key=OPENAI_API_KEY)
    retriever = docsearch.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.session_state.chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, max_tokens_limit=4000)
    st.session_state.chat_active = True  # Enable chat interface
    user_input = query.strip()  # Ensure whitespace isn't causing issues
    chat_history = []
    if user_input:
        chat_history.append({'role': 'user', 'content': user_input})

                    # Compose the conversation into a single string for processing
        conversation = [f"{item['role']}: {item['content']}" for item in chat_history]
                    # max_context_length = 4096  # Adjust based on the model's maximum token limit
                    # combined_conversation = "\n".join(conversation)[-max_context_length:]

                    # Get response from the conversational model
        response = st.session_state.chain.run({'question': user_input})

                    # Append model response to chat history
        chat_history.append({'role': 'assistant', 'content': response})

                    # Clear the input box after processing
        st.session_state.chat_input = ''

                    # Optionally, truncate chat history to manage token size
                    # Keep only the last 10 exchanges
        if len(chat_history) > 20:
           chat_history = chat_history[-20:]
    return response

def generate_document(prompt):
    # Load the data from the CSV file
    df = pd.read_csv('Financial-Document-Generation-using-Retrieval-Augmented-Generation/Raw Data/Business Domains and Financial Docs.csv')

    # Split the prompt into individual words
    prompt_words = prompt.lower().strip().split()

    # Find the best match for the prompt
    best_match = None
    best_match_score = 0

    for index, row in df.iterrows():
        document = row['Document'].lower()
        domain = row['Domain'].lower()
        key_aspects = [x.lower() for x in str(row['Key_Aspects']).split(', ')]

        # Calculate the match score based on the number of matching words
        match_score = sum(1 for word in prompt_words if word in document.split() or word in domain.split() or any(word in aspect.split() for aspect in key_aspects))

        if match_score > best_match_score:
            best_match = row
            best_match_score = match_score

    if best_match is not None:
        llm = OpenAI(temperature=0.7, max_tokens=1024, openai_api_key=OPENAI_API_KEY)
        response = llm(f"Generate a {best_match['Document']} for {best_match['Domain']} for {best_match['Key_Aspects']}")
        return response
    else:
        # If no match is found, return an error message with the list of allowed document types
        allowed_documents = df['Document'].unique()
        error_message = "I'm sorry, I don't have enough knowledge to generate that type of document. Please choose from the following list of allowed document types:\n\n"
        for doc in allowed_documents:
            error_message += "- " + doc + "\n"
        return error_message

def save_note(content):
    if 'note_saved' not in st.session_state:
        st.session_state['note_saved'] = False
    st.session_state['generated_content'] = content

# Streamlit app
def main():
    st.title("Financial Document QnA bot")

    # Sidebar menu
    st.sidebar.title("Menu")
    menu_choice = st.sidebar.selectbox("Choose an option", ["Search Documents", "Generate your own document"])

    if menu_choice == "Search Documents":
        st.subheader("Search Documents")
        query = st.text_input("Enter your query:")
        if st.button("Search"):
            if query:
                st.write("Querying for documents...")
                try:
                    response = semantic_search(query)
                    st.write("Response:", response)
                    save_note(response)
                except ValueError as e:
                    st.write("Error:", e)
                    st.write("The vector store is empty. Please upload documents first.")
            else:
                st.warning("Please enter a query.")

    elif menu_choice == "Generate your own document":
        st.subheader("Generate your own document")
        prompt = st.text_area("Enter your prompt:")
        if st.button("Generate"):
            if prompt:
                st.write("Generating document...")
                try:
                    response = generate_document(prompt)
                    st.write("Generated document:", response)
                    save_note(response)
                except Exception as e:
                    st.write("Error:", e)
            else:
                st.warning("Please enter a prompt.")

        if 'generated_content' in st.session_state and not st.session_state.get('note_saved', True):
            if st.button("Save Note"):
                with open('note.txt', 'a') as file:
                    file.write(st.session_state['generated_content'] + '\n\n')
                st.session_state['note_saved'] = True
                st.success('Note saved successfully!')

if __name__ == "__main__":
    main()