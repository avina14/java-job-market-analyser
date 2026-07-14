import pandas as pd
import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_ADMIN_KEY")
if not api_key:
    raise RuntimeError("Missing OpenAI API key. Set OPENAI_API_KEY in your .env file or export it in the shell.")

client = OpenAI(api_key=api_key)
# Load cleaned data — use only 500 rows to save API costs
df = pd.read_csv("data/cleaned_postings.csv")
df = df.head(500)

function_definition = [{
    "type": "function",
    "function": {
        "name": "extract_job_details",
        "description": "Extract structured information from a job posting description",
        "parameters": {
            "type": "object",
            "properties": {
                "skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of technical skills mentioned e.g. Java, Python, Kafka, Kubernetes"
                },
                "experience_years": {
                    "type": "string",
                    "description": "Years of experience required e.g. 3-5 years"
                },
                "seniority": {
                    "type": "string",
                    "description": "Seniority level: Junior, Mid, Senior, Lead, Principal"
                },
                "remote": {
                    "type": "string",
                    "description": "Remote policy: Remote, Hybrid, On-site"
                }
            }
        }
    }
}]

results = []
for index, row in df.iterrows():
    try:
        messages = [
            {"role": "system", "content": "You are a technical recruiter. Extract structured data from job postings."},
            {"role": "user", "content": f"Extract details from this job posting:\nTitle: {row['title']}\nDescription: {str(row['description'])[:1000]}"}
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=function_definition,
        )
        data = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        data['title'] = row['title']
        data['company_name'] = row['company_name']
        data['location'] = row['location']
        data['normalized_salary'] = row['normalized_salary']
        results.append(data)

        if index % 50 == 0:
            print(f"✅ Processed {index} jobs...")

    except Exception as e:
        print(f"❌ Error at row {index}: {e}")
        continue

# Save results
df_extracted = pd.DataFrame(results)
df_extracted.to_csv("data/extracted_jobs.csv", index=False)
print(f"\n✅ Done! Extracted {len(df_extracted)} jobs saved to data/extracted_jobs.csv")
print(df_extracted.head(3))