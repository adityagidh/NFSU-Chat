import os
# from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM

embeddings = OllamaEmbeddings(model="llama3.2")

vector_db = Chroma(embedding_function=embeddings, collection_name="nfsuchat", persist_directory="./chroma_db")

def add_document_to_chroma(file_path):
    loader = TextLoader(file_path, encoding="utf-8")
    doc = loader.load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
    text = text_splitter.split_documents(doc)
    vector_db.add_documents(text)
    
    print("Chunks added")

def main():
    while True:
        file_path = input("Enter document path: ")

        if file_path.lower() == "q":
            break
        if os.path.exists(file_path):
            add_document_to_chroma(file_path)
        else:
            print("file not found")
            
if __name__ == "__main__":
    main()
