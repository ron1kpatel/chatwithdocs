import argparse
import os
import time
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Load the GROQ and OpenAI API KEY
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize ChatGroq
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt_template = """
Analyze and understand context in detailed and accurately,
Based on the provided context, please provide a detailed and comprehensive answer to the following question. Include relevant examples and explanations.
Context: {context}
Question: {input}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

def vector_embedding(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)
    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors

def main(docs_paths, query):
    all_docs = []
    for doc_path in docs_paths:
        loader = PyPDFLoader(doc_path)  # Load each document
        docs = loader.load()  # Document Loading
        all_docs.extend(docs)  # Combine documents

    vectors = vector_embedding(all_docs)  # Vector OpenAI embeddings
    
    # Create the document chain and retrieval chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start = time.process_time()
    response = retrieval_chain.invoke({'input': query})
    print("Response time:", time.process_time() - start)
    print("Answer:", response['answer'])
    
    # print("\nDocument Similarity Search:")
    # for i, doc in enumerate(response["context"]):
    #     print(f"Document {i+1}:\n", doc.page_content)
    #     print("--------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with Documents CLI Tool")
    parser.add_argument("--docs", nargs='+', required=True, help="List of PDF documents to analyze")
    parser.add_argument("--query", required=True, help="Question to ask about the documents")
    
    args = parser.parse_args()
    
    main(args.docs, args.query)
