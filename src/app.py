import os
import streamlit as st
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
from dotenv import load_dotenv

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Java Job Market Analyser",
    page_icon="🔍",
    layout="wide"
)

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_csvs():
    skill_freq = pd.read_csv("data/skill_frequency.csv", index_col=0)
    seniority_salary = pd.read_csv("data/seniority_salary_summary.csv")
    jobs = pd.read_csv("data/extracted_jobs.csv")
    return skill_freq, seniority_salary, jobs

skill_freq, seniority_salary, jobs_df = load_csvs()

# --------------------------------------------------
# LOAD CHROMA
# --------------------------------------------------
def build_chroma_if_needed():
    chroma_path = "data/chroma_db"
    if os.path.exists(chroma_path):
        return  # already built

    st.info("⚙️ Building vector database for first time... (takes ~2 mins)")
    
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    
    try:
        chroma_client.delete_collection("job_postings")
    except:
        pass
    
    collection = chroma_client.create_collection(
        name="job_postings",
        embedding_function=embedding_fn
    )
    
    df = pd.read_csv("data/extracted_jobs.csv")
    df = df.dropna(subset=['title', 'company_name']).reset_index(drop=True)
    
    def build_chunk(row):
        salary = f"${row['normalized_salary']:,.0f}" if pd.notna(row['normalized_salary']) else "Not specified"
        return f"""Job Title: {row['title']}
Company: {row['company_name']}
Location: {row['location']}
Seniority: {row['seniority']}
Remote Policy: {row['remote']}
Experience Required: {row['experience_years']}
Salary: {salary}
Skills: {row['skills']}""".strip()
    
    df['chunk'] = df.apply(build_chunk, axis=1)
    
    BATCH_SIZE = 50
    for i in range(0, len(df), BATCH_SIZE):
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
    st.success("✅ Vector database ready!")
    st.rerun()

build_chroma_if_needed()
@st.cache_resource
def load_chroma():
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )

    chroma_client = chromadb.PersistentClient(
        path="data/chroma_db"
    )

    collection = chroma_client.get_collection(
        name="job_postings",
        embedding_function=embedding_fn
    )

    return collection

collection = load_chroma()

# --------------------------------------------------
# CHAT MEMORY
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🔍 Java Job Market Analyser")
st.markdown(
    "AI-powered analysis of software engineering job postings using "
    "OpenAI, ChromaDB, RAG and LinkedIn job data."
)

# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "📊 Dashboard",
    "🔎 Semantic Search",
    "🤖 Chatbot"
])

# ==================================================
# TAB 1 DASHBOARD
# ==================================================

with tab1:

    st.subheader("Market Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Jobs Analysed", len(jobs_df))
    col2.metric("Unique Skills", len(skill_freq))
    col3.metric("Top Skill", "Java")
    col4.metric("Median Salary", "$116,500")

    st.divider()

    st.subheader("Top Skills")

    st.bar_chart(skill_freq.head(20))

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Seniority Distribution")
        st.bar_chart(
            jobs_df["seniority"].value_counts()
        )

    with col2:
        st.subheader("Remote Policy")
        st.bar_chart(
            jobs_df["remote"].value_counts()
        )

    st.divider()

    st.subheader("Median Salary by Seniority")

    salary_data = (
        seniority_salary
        .dropna(subset=["median"])
        .sort_values("median", ascending=False)
        .set_index("seniority")
    )

    st.bar_chart(salary_data["median"])

    st.divider()

    st.subheader("Browse Jobs")

    search = st.text_input(
        "Search jobs",
        key="dashboard_search"
    )

    filtered = jobs_df

    if search:
        filtered = jobs_df[
            jobs_df["title"].str.contains(
                search,
                case=False,
                na=False
            )
            |
            jobs_df["company_name"].str.contains(
                search,
                case=False,
                na=False
            )
        ]

    st.dataframe(
        filtered[
            [
                "title",
                "company_name",
                "location",
                "seniority",
                "remote",
                "normalized_salary"
            ]
        ].head(100),
        use_container_width=True
    )

# ==================================================
# TAB 2 SEMANTIC SEARCH
# ==================================================

with tab2:

    st.subheader("Semantic Job Search")

    query = st.text_input(
        "Describe the job you're looking for",
        placeholder="Senior Java engineer with Kafka and AWS experience"
    )

    if query:

        results = collection.query(
            query_texts=[query],
            n_results=5
        )

        docs = results["documents"][0]

        st.success(
            f"Found {len(docs)} similar jobs"
        )

        for i, doc in enumerate(docs):

            with st.expander(
                f"Result {i+1}",
                expanded=(i == 0)
            ):
                st.text(doc)

# ==================================================
# TAB 3 CHATBOT
# ==================================================

with tab3:

    st.subheader("RAG Job Market Assistant")

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input(
        "Ask about salaries, skills, companies..."
    )

    if user_input:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        # ----------------------------
        # Retrieve
        # ----------------------------

        results = collection.query(
            query_texts=[user_input],
            n_results=5
        )

        context = "\n\n".join(
            results["documents"][0]
        )

        memory = st.session_state.messages[-8:]

        messages = [
            {
                "role": "system",
                "content": f"""
You are a software engineering job market analyst.

Answer only using the information below.

Context:
{context}
"""
            }
        ]

        messages.extend(memory)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500
        )

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        with st.chat_message("assistant"):
            st.markdown(answer)