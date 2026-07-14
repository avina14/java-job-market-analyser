import pandas as pd
import ast
from collections import Counter

pd.set_option("display.max_rows", 30)
pd.set_option("display.width", 120)

# ---------------------------------------------------------
# Load extracted data (output of Step 1)
# ---------------------------------------------------------
df = pd.read_csv("data/extracted_jobs.csv")
print(f"Loaded {len(df)} rows\n")


# ---------------------------------------------------------
# Clean up
# ---------------------------------------------------------
def parse_skills(val):
    """The 'skills' column was saved as a stringified list e.g. "['Java', 'Kafka']".
    Convert it back into an actual Python list, safely."""
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    try:
        parsed = ast.literal_eval(val)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


df["skills_list"] = df["skills"].apply(parse_skills)

# normalized_salary should be numeric; coerce anything unparsable to NaN
df["normalized_salary"] = pd.to_numeric(df["normalized_salary"], errors="coerce")

# tidy up text fields so grouping doesn't split on casing/whitespace
for col in ["seniority", "remote", "company_name", "location"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# normalize casing so "Not specified" / "not specified" etc. don't split into
# separate buckets — title-case everything, then fold obvious remote-policy variants
df["seniority"] = df["seniority"].str.title()
df["remote"] = df["remote"].str.title().replace({
    "100% Remote": "Remote",
    "Onsite/Hybrid": "Hybrid",
    "Flexible": "Hybrid",
})

# collapse stray skill-casing duplicates (e.g. "Project Management" vs "project management")
# without mangling correctly mixed-case names like "JavaScript" or "AutoCAD" —
# keep the FIRST-seen casing as the canonical display form for each lowercase key
_skill_canonical = {}
def normalize_skills(skills):
    cleaned = []
    for s in skills:
        s = s.strip()
        if not s:
            continue
        key = s.lower()
        if key not in _skill_canonical:
            _skill_canonical[key] = s  # remember first-seen casing
        cleaned.append(_skill_canonical[key])
    return cleaned

df["skills_list"] = df["skills_list"].apply(normalize_skills)

# drop implausible salaries (e.g. hourly rates like $22 mistakenly stored as annual)
MIN_PLAUSIBLE_SALARY = 1000
df.loc[df["normalized_salary"] < MIN_PLAUSIBLE_SALARY, "normalized_salary"] = pd.NA

n_missing_salary = df["normalized_salary"].isna().sum()
print(f"Note: {n_missing_salary}/{len(df)} rows have no usable salary data ({n_missing_salary/len(df):.0%})\n")


# ---------------------------------------------------------
# 1. Skill frequency
# ---------------------------------------------------------
print("=" * 60)
print("TOP 20 SKILLS")
print("=" * 60)

all_skills = [skill.strip() for skills in df["skills_list"] for skill in skills if skill.strip()]
skill_counts = Counter(all_skills)
skill_freq = pd.Series(skill_counts).sort_values(ascending=False)
print(skill_freq.head(20))


# ---------------------------------------------------------
# 2. Salary ranges
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("SALARY OVERVIEW")
print("=" * 60)

salary_stats = df["normalized_salary"].describe()
print(salary_stats)

print("\nMedian salary by seniority:")
print(
    df.groupby("seniority")["normalized_salary"]
    .agg(["median", "mean", "count"])
    .sort_values("median", ascending=False)
)


# ---------------------------------------------------------
# 3. Top companies (by job postings, and by median salary)
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("TOP 15 COMPANIES BY POSTING COUNT")
print("=" * 60)
top_companies = df["company_name"].value_counts().head(15)
print(top_companies)

print("\nTop 15 companies by median salary (min 3 postings):")
company_salary = (
    df.groupby("company_name")["normalized_salary"]
    .agg(["median", "count"])
    .query("count >= 3")
    .sort_values("median", ascending=False)
    .head(15)
)
print(company_salary)


# ---------------------------------------------------------
# 4. Bonus: seniority + remote distribution
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("SENIORITY DISTRIBUTION")
print("=" * 60)
print(df["seniority"].value_counts())

print("\n" + "=" * 60)
print("REMOTE POLICY DISTRIBUTION")
print("=" * 60)
print(df["remote"].value_counts())


# ---------------------------------------------------------
# Save summary tables for Step 4 (Streamlit dashboard)
# ---------------------------------------------------------
skill_freq.head(30).to_csv("data/skill_frequency.csv", header=["count"])
company_salary.to_csv("data/company_salary_summary.csv")
df.groupby("seniority")["normalized_salary"].agg(["median", "mean", "count"]).to_csv(
    "data/seniority_salary_summary.csv"
)

print("\n✅ Analysis complete. Summary CSVs saved to data/ for the dashboard step.")