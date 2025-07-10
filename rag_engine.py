import os
import fitz
import re
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model_gemini = genai.GenerativeModel("gemini-1.5-flash")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.get_or_create_collection(name="medical_papers")


def preprocess_text(text, chunk_size=1000, overlap=100):
    cleaned = re.sub(r"\s+", " ", text).strip()
    chunks = []
    for i in range(0, len(cleaned), chunk_size - overlap):
        chunk = cleaned[i:i + chunk_size]
        if len(chunk) > 100:
            chunks.append(chunk)
    return chunks


def extract_chunks_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    for i, page in enumerate(doc):
        text = page.get_text()
        page_chunks = preprocess_text(text)
        for chunk in page_chunks:
            chunks.append((chunk, i + 1))
    return chunks


def store_chunks(chunks, file_name):
    for idx, (chunk, page_no) in enumerate(chunks):
        embedding = embedding_model.encode([chunk])
        metadata = {'source': file_name, 'page': page_no}
        collection.add(
            documents=[chunk],
            embeddings=embedding,
            metadatas=[metadata],
            ids=[f"{file_name}_p{page_no}_c{idx}"]
        )


def process_pdfs(data_folder):
    pdf_files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        path = os.path.join(data_folder, pdf_file)
        chunks = extract_chunks_from_pdf(path)
        store_chunks(chunks, pdf_file)


def retrieve_context(query, k=5):
    query_embedding = embedding_model.encode([query])
    results = collection.query(query_embeddings=query_embedding, n_results=k)
    docs = [doc for sublist in results['documents'] for doc in sublist]
    return docs


def get_gemini_response(query, context_chunks):
    context = "\n\n".join(context_chunks[:3])
    prompt = (
        f"You are a helpful medical assistant. Use the following documents:\n\n"
        f"{context} \n\n"
        f"Question: {query}\nAnswer:"
    )
    response = model_gemini.generate_content([prompt])
    return response.text.strip()
