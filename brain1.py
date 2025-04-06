import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
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






openai_api_key = "key"



#iterate over all files uploaded to website

client = OpenAI(api_key=openai_api_key)



#file_name = xxxx

def load_file(file_names):


    all_docs = []


    supported_extensions = ['.py', '.java', '.js', '.cpp', '.html', '.css', '.ts', '.rb', '.go', '.php']


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

        elif extension == ".jpg" or extension == ".webp" or extension == ".jpeg" or extension == ".png" or extension == ".gif" or  extension == ".bmp" or extension == ".gif":
            textt = extract_text_from_image(file_name)
            loader = TextLoader(textt)

        else:
            raise ValueError(f"Unsupported file type: {extension}")

        doc = loader.load()

        for d in doc:
            d.metadata = metadata



        all_docs.extend(doc)

    return all_docs







def extract_text_from_image(imageurl):
    
    response = requests.get(imageurl)
    image = Image.open(BytesIO(response.content))



    text = pytesseract.image_to_string(image)

    return text





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

    print(relevant_data[0].metadata)

    gpt_prompt = f"Answer the question using the context provided: \n\n {context} \n\n Question: {question}"


    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": gpt_prompt}
        ],
        max_tokens=200
    )

    return response.choices[0].message.content



file_paths = [
    r"C:\Users\anees\Downloads\Dantuluri, Aneesh - Transcript  (1).pdf",
    r"C:\Users\anees\Downloads\Event Liability Release + Medical Authorization (4).pdf",
    r"C:\Users\anees\Downloads\graphs_worksheet1_java_aplus.pdf",
    r"C:\Users\anees\Downloads\RMP Personal Statement.pdf",
    r"C:\Users\anees\PycharmProjects\toneFinda\spotifyy.py"
    ]

fil = load_file(file_paths)
print("File loaded successfully")


vec_file = embed_file(fil)
print("File loaded and embedded successfully")

q = search("What is the saying related to birds in my data", vec_file)



outp = generate_answer(q, "What is the saying related to birds in my data")

print(outp)
print("Answer generated successfully")
