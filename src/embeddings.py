import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
import os

load_dotenv()

# ── Setup ChromaDB ──
embedding_fn = OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

client = chromadb.PersistentClient(path="data/chroma_db")

# Delete collection if exists (fresh start)
try:
    client.delete_collection("job_postings")
except:
    pass

collection = client.create_collection(
    name="job_postings",
    embedding_function=embedding_fn
)

# ── Load extracted jobs ──
df = pd.read_csv("data/extracted_jobs.csv")
df = df.dropna(subset=['title', 'company_name'])
df = df.reset_index(drop=True)

print(f"Loaded {len(df)} jobs")

# ── Chunk each job into a meaningful text chunk ──
def build_chunk(row):
    skills = row['skills'] if isinstance(row['skills'], str) else "Not specified"
    salary = f"${row['normalized_salary']:,.0f}" if pd.notna(row['normalized_salary']) else "Not specified"
    return f"""
Job Title: {row['title']}
Company: {row['company_name']}
Location: {row['location']}
Seniority: {row['seniority']}
Remote Policy: {row['remote']}
Experience Required: {row['experience_years']}
Salary: {salary}
Skills: {skills}
""".strip()

df['chunk'] = df.apply(build_chunk, axis=1)

# ── Add to ChromaDB in batches of 50 ──
BATCH_SIZE = 50
total = len(df)

for i in range(0, total, BATCH_SIZE):
    batch = df.iloc[i:i+BATCH_SIZE]
    collection.add(
        ids=[str(idx) for idx in batch.index],
        documents=batch['chunk'].tolist(),
        metadatas=[{
            'title': str(row['title']),
            'company': str(row['company_name']),
            'location': str(row['location']),
            'seniority': str(row['seniority']),
            'remote': str(row['remote']),
            'salary': str(row['normalized_salary'])
        } for _, row in batch.iterrows()]
    )
    print(f"✅ Embedded {min(i+BATCH_SIZE, total)}/{total} jobs...")

print(f"\n✅ Done! {collection.count()} job chunks stored in ChromaDB")

# ── Test semantic search ──
print("\n🔍 Test search: 'Senior Java engineer with Kafka experience'")
results = collection.query(
    query_texts=["Senior Java engineer with Kafka experience"],
    n_results=3
)
for i, doc in enumerate(results['documents'][0]):
    print(f"\n--- Result {i+1} ---")
    print(doc)