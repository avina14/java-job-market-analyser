import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
from dotenv import load_dotenv

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
print("\n🤖 Java Job Market Analyser Chatbot")
print("Ask me anything about the job market!")
print("Type 'quit' to exit\n")

# ── Conversation memory — last 4 interactions ──
memory = []

def chat(user_message):
    # Step 1 — Semantic search for relevant jobs
    results = collection.query(
        query_texts=[user_message],
        n_results=5
    )
    context = "\n\n".join(results['documents'][0])

    # Step 2 — Build messages with memory (last 4 interactions)
    messages = [
        {
            "role": "system",
            "content": f"""You are a job market analyst specialising in software engineering roles.
Answer questions using the job data provided below.
Be specific, cite companies and salaries where available.
If the data doesn't contain enough info, say so honestly.

Relevant job postings from our database:
{context}"""
        }
    ]

    # Add last 4 interactions from memory
    messages.extend(memory[-8:])  # 4 interactions = 8 messages (user+assistant)

    # Add current user message
    messages.append({"role": "user", "content": user_message})

    # Step 3 — Get response
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=500
    )

    assistant_reply = response.choices[0].message.content

    # Step 4 — Update memory
    memory.append({"role": "user", "content": user_message})
    memory.append({"role": "assistant", "content": assistant_reply})

    return assistant_reply

# ── Main chat loop ──
while True:
    user_input = input("You: ").strip()
    if user_input.lower() == 'quit':
        print("Goodbye!")
        break
    if not user_input:
        continue
    response = chat(user_input)
    print(f"\n🤖 Bot: {response}\n")