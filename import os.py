import os
import faiss
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from collections import Counter

# ----------------- Utility functions -----------------

def exact_match(prediction, ground_truth):
    return int(prediction.strip().lower() == ground_truth.strip().lower())

def f1_score(prediction, ground_truth):
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

# ----------------- Load and index documents -----------------

def load_documents_from_folder(folder_path):
    docs = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                docs.append(content)
                filenames.append(filename)
    return docs, filenames

def build_faiss_index(documents, embedder):
    embeddings = embedder.encode(documents, convert_to_tensor=True).cpu().detach().numpy()
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def retrieve_documents(index, embedder, query, documents, k=3):
    query_emb = embedder.encode([query], convert_to_tensor=True).cpu().detach().numpy()
    distances, indices = index.search(query_emb, k)
    retrieved = [documents[i] for i in indices[0]]
    return retrieved

# ----------------- Models and generation -----------------

class RAGModel:
    def __init__(self):
        print("Loading tokenizer and generative model...")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        self.generator = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("Models loaded.")
        
    def generate_text(self, prompt, max_length=60):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.generator.generate(**inputs, max_length=max_length, num_beams=5, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def simple_rag(self, query, index, documents, k=3):
        retrieved_docs = retrieve_documents(index, self.embedder, query, documents, k)
        context = query + " " + " ".join(retrieved_docs)
        answer = self.generate_text(context)
        return answer, retrieved_docs
    
    def hyde(self, query, index, documents, k=3):
        hypothetical_doc = self.generate_text(query, max_length=100)
        retrieved_docs = retrieve_documents(index, self.embedder, hypothetical_doc, documents, k)
        context = query + " " + hypothetical_doc + " " + " ".join(retrieved_docs)
        answer = self.generate_text(context)
        return answer, hypothetical_doc, retrieved_docs

# ----------------- Batch processing -----------------

def batch_evaluate(questions_df, rag_model, index, documents, k=3):
    results = []
    
    for i, row in questions_df.iterrows():
        query = row['question']
        ground_truth = row['answer']
        
        ans_simple, _ = rag_model.simple_rag(query, index, documents, k)
        ans_hyde, hypo_doc, _ = rag_model.hyde(query, index, documents, k)
        
        em_simple = exact_match(ans_simple, ground_truth)
        f1_simple = f1_score(ans_simple, ground_truth)
        
        em_hyde = exact_match(ans_hyde, ground_truth)
        f1_hyde = f1_score(ans_hyde, ground_truth)
        
        better = "Equal"
        if f1_hyde > f1_simple:
            better = "HyDE"
        elif f1_simple > f1_hyde:
            better = "Simple RAG"
        
        results.append({
            'question': query,
            'ground_truth': ground_truth,
            'simple_rag_answer': ans_simple,
            'hyde_answer': ans_hyde,
            'hyde_hypothetical_doc': hypo_doc,
            'simple_rag_em': em_simple,
            'simple_rag_f1': f1_simple,
            'hyde_em': em_hyde,
            'hyde_f1': f1_hyde,
            'better_method': better
        })
        
        print(f"Processed {i+1}/{len(questions_df)} questions", end='\r')
    
    return pd.DataFrame(results)

# ----------------- Main -----------------

def main():
    print("Enter the path to your documents folder (txt files):")
    folder_path = input().strip()
    
    print("Loading documents...")
    documents, filenames = load_documents_from_folder(folder_path)
    if len(documents) == 0:
        print("No txt documents found in the folder.")
        return
    
    print(f"Loaded {len(documents)} documents.")
    
    rag_model = RAGModel()
    
    print("Building FAISS index...")
    index, _ = build_faiss_index(documents, rag_model.embedder)
    print("Index built.")
    
    print("Enter the path to your CSV file with questions and answers (columns: question, answer):")
    csv_path = input().strip()
    
    questions_df = pd.read_csv(csv_path)
    
    if 'question' not in questions_df.columns or 'answer' not in questions_df.columns:
        print("CSV must contain 'question' and 'answer' columns.")
        return
    
    print(f"Loaded {len(questions_df)} questions.")
    
    print("Starting batch evaluation...")
    results_df = batch_evaluate(questions_df, rag_model, index, documents, k=3)
    
    output_csv = "rag_comparison_results.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"Batch evaluation completed. Results saved to {output_csv}")
    
    # Summary statistics
    simple_better = sum(results_df['better_method'] == 'Simple RAG')
    hyde_better = sum(results_df['better_method'] == 'HyDE')
    equal = sum(results_df['better_method'] == 'Equal')
    
    print(f"\nSummary:")
    print(f"Simple RAG performed better on {simple_better} questions.")
    print(f"HyDE performed better on {hyde_better} questions.")
    print(f"Both performed equally on {equal} questions.")

if __name__ == "__main__":
    main()
