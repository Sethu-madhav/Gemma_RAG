import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines import pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- optimization : Enable TensorFloat32 for better performance
torch.set_float32_matmul_precision('high')

# --- 1. SET UP MODEL AND TOKENIZER ---
print("--- Setting up model and tokenizer ---")
model_id = "google/gemma-3-4b-it"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype = torch.bfloat16,
    device_map="auto",
    token=True 
)
tokenizer = AutoTokenizer.from_pretrained(model_id, token=True) # Added token=True for authentication
print("Model and tokenizer setup complete.")

# --- 2. CREATE A TEXT GENERATION PIPELINE ---
print("\n--- Creating text generation pipeline ---")
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    top_p=0.95,
    temperature=0.1,
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
print("Text generation pipeline created.")


# --- Main Execution Guard ---
if __name__ == "__main__":
    print("\n--- Starting RAG pipeline build ---")

    # --- 3. LOAD DOCUMENTS ---
    print("--- Loading documents ---")
    documents_dir = './documents/'

    # Load docs
    loader = DirectoryLoader(documents_dir, show_progress=True, use_multithreading=True)
    documents = loader.load()
    
    if not documents:
        print("No documents found. Please add .txt or .pdf files to the 'documents' folder.")
        exit() # Exit if no documents are found
    
    print(f"Loaded {len(documents)} document(s).")

    # --- 4. SPLIT DOCUMENTS INTO CHUNKS ---
    print("--- Splitting documents into chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(documents)
    print(f"Split documents into {len(all_splits)} chunks.")

    # --- 5. CREATE EMBEDDINGS AND VECTOR STORE ---
    print("--- Creating embeddings and vector store ---")
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    print(f"Model {model_name} is loaded.")
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    vectorstore = FAISS.from_documents(all_splits, embeddings)
    print("Vector store created successfully.")

    # --- 6. CREATE THE RETRIEVALQA CHAIN (REVISED LOGIC) ---
    print("--- Creating RetrievalQA chain ---")
    template = """<start_of_turn>user
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use five sentences maximum and keep the answer as concise as possible.
    
    CONTEXT: 
    {context}

    QUESTION: 
    {question}<end_of_turn>
    <start_of_turn>model
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    
    # Create the retriever object directly from the vector store
    retriever = vectorstore.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=retriever, # Pass the retriever object
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True
    )
    print("RetrievalQA chain created successfully.")
    print("\n--- RAG Pipeline is ready! ---")

    # --- 7. Create Interactive CLI ---
    print("Enter 'exit' or 'quit to stop the application.")
    while True:
        # Propt the user for a question
        query = input("\nAsk a question: ")

        if query.lower() in ['exit', 'quit']:
            print("Exiting application. Goodbye!")
            break
        
        # Run the RAG chain with the user's question
        print("Thinking...")
        result = qa_chain.invoke({"query": query})

        # Print the answer
        print("\nAnswer:")
        print(result["result"].strip())
