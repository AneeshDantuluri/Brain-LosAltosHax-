import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import 


UnstructuredMarkdownLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import Docx2txtLoader
from openai import OpenAI
import pytesseract
from PIL import Image
import requests
from PIL import Image
import pytesseract
from io import BytesIO
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
import datetime
import firebase_admin
from firebase_admin import credentials, storage
from flask import Flask, request, jsonify
from flask_cors import CORS

# Firebase initialization
cred = credentials.Certificate("redacted")
firebase_admin.initialize_app(cred, {"storageBucket": "redacted"})

openai_api_key = "redacted"
client = OpenAI(api_key=openai_api_key)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})
# Helper functions
def extract_text_from_image(imageurl):
    response = requests.get(imageurl)
    image = Image.open(BytesIO(response.content))
    text = pytesseract.image_to_string(image)
    return text

def load_file(file_names):
    all_docs = []
    supported_extensions = ['.py', '.java', '.js', '.cpp', '.html', '.css', '.ts', '.rb', '.go', '.php', '.vue']

    for file_name in file_names:
        extension = os.path.splitext(file_name)[1]

        metadata = {
            "file_name": file_name.split("\\")[-1],
            "file_path": file_name,
            "file_size": os.path.getsize(file_name),
            "created_at": datetime.datetime.fromtimestamp(os.path.getctime(file_name)),
            "modified_at": datetime.datetime.fromtimestamp(os.path.getmtime(file_name)),
        }

        if extension == ".txt":
            loader = TextLoader(file_name)
        elif extension == ".pdf":
            loader = PyPDFLoader(file_name)
        elif extension == ".docx":
            loader = Docx2txtLoader(file_name)
        elif extension == ".csv":
            loader = CSVLoader(file_name)
        elif extension == ".json":
            loader = JSONLoader(file_name)
        elif extension == ".html":
            loader = UnstructuredHTMLLoader(file_name)
        elif extension == ".md":
            loader = UnstructuredMarkdownLoader(file_name)
        elif extension == ".xml":
            loader = TextLoader(file_name)
        elif extension == ".xlsx":
            loader = UnstructuredExcelLoader(file_name)
        elif extension in supported_extensions:
            loader = TextLoader(file_name)
        elif extension in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
            textt = extract_text_from_image(file_name)
            loader = TextLoader(textt)
        else:
            raise ValueError(f"Unsupported file type: {extension}")

        doc = loader.load()

        for d in doc:
            d.metadata = metadata

        all_docs.extend(doc)

    return all_docs

def embed_file(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def search(query, vectorstore):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    query_vector = embeddings.embed_query(query)
    return vectorstore.similarity_search_by_vector(query_vector, k=5)

def generate_answer(relevant_data, question):
    context = "\n".join([doc.page_content for doc in relevant_data])

    # Get the metadata of the file from which the answer vector was retrieved
    metadata = relevant_data[0].metadata if relevant_data else None
    file_name = metadata['file_name'] if metadata else 'No file found'

    gpt_prompt = f"Answer the question using the context provided with enough detail (your a helpful AI assistant): \n\n {context} \n\n Question: {question}"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": gpt_prompt}],
        max_tokens=200
    )

    # Return the generated answer along with the metadata (file name)
    return response.choices[0].message.content, file_name

# Firebase functions
def download_files_from_firebase():
    # Get files from Firebase storage
    bucket = storage.bucket()
    blobs = bucket.list_blobs()
    file_names = []
    
    for blob in blobs:
        file_path = "/tmp/" + blob.name.split("/")[-1]
        blob.download_to_filename(file_path)
        file_names.append(file_path)
    
    return file_names

# API Endpoints
@app.route('/search', methods=['GET'])
def search_endpoint():
    query = request.args.get('query')

    # Download files from Firebase and load them
    file_paths = download_files_from_firebase()
    docs = load_file(file_paths)
    
    # Update vector database
    vec_file = embed_file(docs)

    # Search for the query
    relevant_data = search(query, vec_file)

    # Generate the answer and get the file from which it was generated
    answer, file_name = generate_answer(relevant_data, query)

    return jsonify({"answer": answer, "file_name": file_name})

@app.route('/update_vector_db', methods=['POST'])
def update_vector_db():
    # Download files from Firebase and load them
    file_paths = download_files_from_firebase()
    docs = load_file(file_paths)
    
    # Update vector database
    vec_file = embed_file(docs)
    
    return jsonify({"message": "Vector database updated successfully"}), 200

if __name__ == '__main__':
    app.run(debug=True)
