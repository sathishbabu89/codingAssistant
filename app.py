import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from qdrant_client.http.models import VectorParams

# Set your Hugging Face API token here
HUGGINGFACE_API_TOKEN = "your_token_here"  # Replace with your actual token

# Initialize Qdrant client
client = QdrantClient("http://localhost:6333")

# Load embedding model
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Example code snippets
code_snippets = [
    "def add(a, b): return a + b",
    "def subtract(a, b): return a - b",
]

# Generate embeddings and store in Qdrant
embeddings = embedder.encode(code_snippets)

# Create collection with vector configuration (Cosine similarity for float vectors)
client.create_collection(
    collection_name="code_snippets", 
    vectors_config=VectorParams(size=384, distance="Cosine")
)

# Insert the points into Qdrant
points = [PointStruct(id=i, vector=emb, payload={"code": code_snippets[i]}) for i, emb in enumerate(embeddings)]
client.upsert(collection_name="code_snippets", points=points)

# Load the LLM (Mistral-Nemo-Instruct-2407) with API token
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-Nemo-Instruct-2407", 
    model_kwargs={"temperature": 0.5},
    huggingfacehub_api_token=HUGGINGFACE_API_TOKEN  # Pass the token directly
)

# Define the prompt template
template = '''
You are a coding assistant. Help with the following code-related query:
{query}
'''
prompt = PromptTemplate(template=template, input_variables=["query"])

# Create the LLM chain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Function to retrieve code snippets using Qdrant
def retrieve_code_snippets(query):
    query_embedding = embedder.encode([query])[0]
    results = client.search(collection_name="code_snippets", query_vector=query_embedding, limit=3)
    return [hit.payload["code"] for hit in results]

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
