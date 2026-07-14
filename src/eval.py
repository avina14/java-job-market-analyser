import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
import pandas as pd

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Load ChromaDB ──
embedding_fn = OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)
chroma_client = chromadb.PersistentClient(path="data/chroma_db")
collection = chroma_client.get_collection(
    name="job_postings",
    embedding_function=embedding_fn
)
print(f"✅ Loaded {collection.count()} jobs from ChromaDB")

# ── 10 Predefined Q&A pairs ──
qa_pairs = [
    {
        "question": "What are the most in-demand skills for software engineers?",
        "ground_truth": "Java, SQL, Python, JavaScript, AWS, and Azure are among the most demanded skills for software engineers."
    },
    {
        "question": "Which companies are hiring Senior Java engineers?",
        "ground_truth": "Companies like Boomi, NInfo Systems, and others are hiring Senior Java engineers with Spring Boot and Kafka experience."
    },
    {
        "question": "What is the average salary for a Senior software engineer?",
        "ground_truth": "The median salary for a Senior software engineer is around $133,000 per year."
    },
    {
        "question": "Which jobs offer remote work for backend engineers?",
        "ground_truth": "Several companies offer remote backend engineering roles including positions requiring Java, Kafka, and Kubernetes skills."
    },
    {
        "question": "What experience is required for a Lead engineer role?",
        "ground_truth": "Lead engineer roles typically require 8-12 years of experience with strong system design and leadership skills."
    },
    {
        "question": "What cloud platforms are most mentioned in job postings?",
        "ground_truth": "AWS and Azure are the most commonly mentioned cloud platforms in software engineering job postings."
    },
    {
        "question": "What is the salary difference between Junior and Senior engineers?",
        "ground_truth": "Junior engineers earn around $68,000 while Senior engineers earn around $133,000 median salary annually."
    },
    {
        "question": "Which companies offer hybrid work for Java developers?",
        "ground_truth": "Companies like Boomi and NInfo Systems offer hybrid work arrangements for Java developers."
    },
    {
        "question": "What skills are needed for a DevOps or Kubernetes role?",
        "ground_truth": "Kubernetes, Docker, Jenkins, Linux, and cloud platforms like AWS are commonly required for DevOps roles."
    },
    {
        "question": "What are the top programming languages required in tech jobs?",
        "ground_truth": "Java, Python, JavaScript, and C# are the top programming languages required across software engineering roles."
    }
]

# ── Run RAG pipeline for each question ──
questions = []
answers = []
contexts = []
ground_truths = []

print("\n🔍 Running RAG pipeline for 10 questions...\n")

for i, qa in enumerate(qa_pairs):
    question = qa["question"]
    ground_truth = qa["ground_truth"]

    # Step 1 — Retrieve context from ChromaDB
    results = collection.query(
        query_texts=[question],
        n_results=3
    )
    retrieved_context = results['documents'][0]

    # Step 2 — Generate answer using OpenAI
    context_text = "\n\n".join(retrieved_context)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""You are a job market analyst. Answer the question using only the context below.
                
Context:
{context_text}"""
            },
            {"role": "user", "content": question}
        ],
        max_tokens=300
    )
    answer = response.choices[0].message.content

    questions.append(question)
    answers.append(answer)
    contexts.append(retrieved_context)
    ground_truths.append(ground_truth)

    print(f"✅ Q{i+1}: {question[:60]}...")

# ── Run RAGAS evaluation ──
print("\n📊 Running RAGAS evaluation...\n")

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)

print("\n" + "="*50)
print("RAGAS EVALUATION RESULTS")
print("="*50)
print(results)

# Save results
results_df = results.to_pandas()
results_df.to_csv("data/ragas_results.csv", index=False)
print("\n✅ Results saved to data/ragas_results.csv")