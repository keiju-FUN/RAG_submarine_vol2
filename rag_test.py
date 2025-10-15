import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")

PDF_FILE_PATH = "your_document.pdf" # ファイル名をここに指定

print(f"Loading {PDF_FILE_PATH}...")
loader = UnstructuredPDFLoader(PDF_FILE_PATH) 
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100) 
texts = text_splitter.split_documents(documents)
print(f"Split into {len(texts)} chunks.")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

print("Creating vector store...")
vectorstore = FAISS.from_documents(texts, embeddings)
print("Vector store created!")

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatGoogleGenerativeAI(model="model_name"), # 有効なモデル名を入力
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

question = "your_question"#質問を入力
print(f"\nQuestion: {question}")
response = qa_chain.invoke(question)

print("\nAnswer:")
print(response['result'])