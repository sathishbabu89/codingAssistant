
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate

# Initialize ChromaDB client
client = chromadb.Client()

# Create a collection for code snippets
collection = client.create_collection("code_snippets")

# Load embedding model
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Example code snippets
code_snippets = [
    "def add(a, b): return a + b",
    "def subtract(a, b): return a - b",
]

# Generate embeddings and store in ChromaDB
embeddings = embedder.encode(code_snippets)
collection.add(
    documents=code_snippets,
    embeddings=embeddings,
    ids=["snippet1", "snippet2"]
)

# Load the LLM (Mistral-Nemo-Instruct-2407)
llm = HuggingFaceHub(repo_id="mistralai/Mistral-Nemo-Instruct-2407", model_kwargs={"temperature": 0.5})

# Define the prompt template
template = '''
You are a coding assistant. Help with the following code-related query:
{query}
'''
prompt = PromptTemplate(template=template, input_variables=["query"])

# Create the LLM chain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Function to retrieve code snippets using ChromaDB
def retrieve_code_snippets(query):
    query_embedding = embedder.encode([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    return results["documents"]

# Function to generate assistant response
def generate_code_assistant_response(query):
    retrieved_snippets = retrieve_code_snippets(query)
    response = llm_chain.run(query=query)
    return response, retrieved_snippets

# Streamlit web app
st.title("Personal Coding Assistant")

query = st.text_input("Enter your coding query:")

if query:
    response, retrieved_snippets = generate_code_assistant_response(query)
    
    st.subheader("LLM Response:")
    st.write(response)
    
    st.subheader("Retrieved Code Snippets:")
    for snippet in retrieved_snippets:
        st.code(snippet)
