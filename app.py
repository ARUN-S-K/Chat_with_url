import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def webpage_text(url_link):
    response = requests.get(url_link)
    soup = BeautifulSoup(response.content, 'html.parser')
    target_divs = soup.find_all('div')

    all_text = ""
    for div in target_divs:
        # Find all target tags within the div
        target_elements=div.find_all('td')
        for element in target_elements:
          text = element.get_text()
          all_text+=text
    soup = BeautifulSoup(all_text, 'html.parser')
    text = soup.get_text(separator='\n', strip=True)

    return text
    
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro")

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    
    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config("Chat With Url")
    st.header("Chat with Url")

    user_question = st.text_input("Ask a Question from the webpage")

    with st.sidebar:
        st.title("Menu:")
        url_link = st.text_input("Enter the webpage url")
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = webpage_text(url_link)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    if user_question:
        user_input(user_question)



if __name__ == "__main__":
    main()
