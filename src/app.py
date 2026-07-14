import streamlit as st
import pandas as pd

st.title("🔍 Java Job Market Analyser")
st.markdown("Analysing 500+ software engineering job postings using OpenAI + LinkedIn data")

# Load data
skill_freq = pd.read_csv("data/skill_frequency.csv", index_col=0)
seniority_salary = pd.read_csv("data/seniority_salary_summary.csv")
df = pd.read_csv("data/extracted_jobs.csv")

# ── Key metrics ──
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Jobs Analysed", len(df))
col2.metric("Unique Skills Found", len(skill_freq))
col3.metric("Top Skill", "Java")
col4.metric("Median Salary (USD)", "$116,500")

st.divider()

# ── Top Skills ──
st.subheader("🔧 Top 20 In-Demand Skills")
st.bar_chart(skill_freq.head(20))

st.divider()

# ── Seniority Distribution ──
col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Seniority Distribution")
    seniority_counts = df['seniority'].value_counts()
    st.bar_chart(seniority_counts)

with col2:
    st.subheader("🏠 Remote Policy")
    remote_counts = df['remote'].value_counts()
    st.bar_chart(remote_counts)

st.divider()

# ── Salary by Seniority ──
st.subheader("💰 Median Salary by Seniority")
salary_data = seniority_salary.dropna(subset=['median'])
salary_data = salary_data.sort_values('median', ascending=False).head(10)
salary_data = salary_data.set_index('seniority')
st.bar_chart(salary_data['median'])

st.divider()

# ── Raw data ──
st.subheader("📋 Job Postings")
search = st.text_input("🔎 Search by title or company")
filtered = df
if search:
    filtered = df[
        df['title'].str.contains(search, case=False, na=False) |
        df['company_name'].str.contains(search, case=False, na=False)
    ]
st.dataframe(
    filtered[['title', 'company_name', 'location', 'seniority', 'remote', 'normalized_salary']].head(50),
    use_container_width=True
)