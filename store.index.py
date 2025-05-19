from src.helper import load_pdf_file, text_split, download_huggingface_embeddings
from pinecone import Pinecone, ServerlessSpec  # ✅ Corrected import
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Load and split documents
extracted_data = load_pdf_file(data='data/')
text_chunks = text_split(extracted_data)

# Load embeddings
embeddings = download_huggingface_embeddings("sentence-transformers/all-MiniLM-L6-v2")


# Define index config
index_name = "test"

# ✅ Create index if it doesn't exist
if index_name not in [index['name'] for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# ✅ Upload documents to Pinecone vector store
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY,
)
