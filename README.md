# 🔍 Java Job Market Analyser

An AI-powered job market analysis tool that ingests 17,000+ software engineering job postings, extracts structured insights using OpenAI function calling, enables semantic search via ChromaDB, and powers a conversational chatbot with memory — evaluated using the RAGAS framework.

## 🎯 Project Highlights
- Analysed **500 job postings** extracted from a 17,000+ LinkedIn dataset
- **OpenAI function calling** to extract skills, seniority, salary, and remote policy
- **ChromaDB vector database** for semantic job search
- **Conversational chatbot** with 4-message memory window
- **RAGAS evaluation** — Faithfulness: 0.83 | Answer Relevancy: 0.98
- **Streamlit dashboard** with interactive charts and job search

## 🏗️ Architecture
LinkedIn CSV (17k jobs)
↓
ingest.py — clean & filter tech jobs
↓
extract.py — OpenAI function calling → extract skills, seniority, salary
↓
analyze.py — pandas insights → skill frequency, salary by seniority
↓
embeddings.py — chunk jobs → embed with text-embedding-3-small → ChromaDB
↓
chatbot.py — semantic retrieval + GPT-4o-mini + 4-message memory
↓
app.py — Streamlit dashboard
↓
eval.py — RAGAS evaluation on 10 predefined Q&A pairs
## 📊 RAGAS Evaluation Results

| Metric | Score |
|---|---|
| Faithfulness | 0.83 |
| Answer Relevancy | 0.98 |
| Context Precision | 0.43 |
| Context Recall | 0.10 |

> Context Recall is low due to the 500-job subset. Architecture is designed to scale to the full 17k dataset.

## 🛠️ Tech Stack

- **Python 3.11**
- **OpenAI API** — GPT-4o-mini (function calling + chat) + text-embedding-3-small
- **ChromaDB** — persistent vector database
- **LangChain** — RAG pipeline
- **Streamlit** — interactive dashboard
- **Pandas** — data analysis
- **RAGAS** — RAG evaluation framework
- **Dataset** — LinkedIn Job Postings 2024 (Kaggle)

## 🚀 Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/java-job-market-analyser.git
cd java-job-market-analyser
```

### 2. Create virtual environment
```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
```

### 3. Add your OpenAI API key
```bash
echo "OPENAI_API_KEY=your_key_here" > .env
```

### 4. Download the dataset
Download [LinkedIn Job Postings 2024](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) from Kaggle and place `postings.csv` in the `data/` folder.

### 5. Run the pipeline
```bash
python src/ingest.py        # Clean and filter jobs
python src/extract.py       # Extract structured data with OpenAI
python src/analyze.py       # Generate insights
python src/embeddings.py    # Build ChromaDB vector store
streamlit run src/app.py    # Launch dashboard
```

### 6. Run the chatbot
```bash
python src/chatbot.py
```

### 7. Run evaluation
```bash
python src/eval.py
```

## 💬 Example Chatbot Interactions

**Q:** What are the most in-demand skills for Java engineers?
**A:** Based on 500 job postings, the top skills are Java, SQL, Python, Spring Boot, AWS, and Kafka...

**Q:** Which companies offer remote Java jobs?
**A:** Companies like Boomi, NInfo Systems, and ZenithMinds offer remote/hybrid Java engineering roles...

**Q:** What salary can I expect as a Senior Java engineer?
**A:** The median salary for Senior software engineers is around $133,000 annually...

## 👩‍💻 Author
**Avina Choudhary** — Senior Software Engineer
- 📧 avina14.ac@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/avinachoudhary)