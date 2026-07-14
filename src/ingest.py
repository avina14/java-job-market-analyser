import pandas as pd

# Load the dataset
df = pd.read_csv("data/postings.csv")

# Keep only useful columns
columns_to_keep = [
    'job_id', 'company_name', 'title', 'description',
    'location', 'formatted_work_type', 'formatted_experience_level',
    'min_salary', 'max_salary', 'normalized_salary', 'remote_allowed'
]
df = df[columns_to_keep]

# Drop rows where description is missing (description is most important)
df = df.dropna(subset=['description'])

# Fill missing values
df['company_name'] = df['company_name'].fillna('Unknown')
df['formatted_experience_level'] = df['formatted_experience_level'].fillna('Not specified')
df['remote_allowed'] = df['remote_allowed'].fillna(0)

# Filter for Software/Tech jobs only
tech_keywords = ['software', 'engineer', 'developer', 'java', 'backend', 
                 'fullstack', 'data', 'devops', 'cloud', 'architect']
mask = df['title'].str.lower().str.contains('|'.join(tech_keywords), na=False)
df_tech = df[mask]

print(f"Total jobs: {len(df)}")
print(f"Tech jobs after filter: {len(df_tech)}")
print(f"\nSample titles:")
print(df_tech['title'].head(10).tolist())
print(f"\nExperience levels:")
print(df_tech['formatted_experience_level'].value_counts())

# Save cleaned data
df_tech.to_csv("data/cleaned_postings.csv", index=False)
print("\n✅ Cleaned data saved to data/cleaned_postings.csv")