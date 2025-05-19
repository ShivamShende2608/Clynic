from flask import Flask, render_template, request
from src.helper import download_huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

# ✅ Flask App Setup
app = Flask(__name__)
load_dotenv()

# ✅ API Keys from .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ✅ Environment variables for LangChain compatibility with Groq
os.environ["OPENAI_API_KEY"] = GROQ_API_KEY
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# ✅ Embedding Model and Retriever
embeddings = download_huggingface_embeddings("sentence-transformers/all-MiniLM-L6-v2")
index_name = "test"
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# ✅ LLM Setup
llm = ChatOpenAI(
    temperature=0.4,
    max_tokens=500,
    model_name="llama3-70b-8192"
)

# ✅ Prompt and Chain setup
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ✅ Landing Page Route
@app.route("/")
def landing():
    return render_template("main.html")  # Ensure main.html is in 'templates/'

# ✅ Chatbot UI Page Route
@app.route("/chat")
def chat_ui():
    return render_template("index.html")  # index.html is chatbot UI

# ✅ Chat API (POST handler)
@app.route("/get", methods=["POST"])
def chat_response():
    msg = request.form["msg"]
    response = rag_chain.invoke({"input": msg})
    return str(response["answer"])

# ✅ Run Server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
