import json
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import re
from datetime import datetime, timedelta

# === OpenAI Setup ===
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
model_id = "gpt-3.5-turbo"

# === Load Embeddings + Incidents ===
with open("incident_texts.json") as f:
    documents = json.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("incident_index.faiss")

# === Streamlit UI ===
st.set_page_config(page_title="Release Incident Advisor Chatbot", layout="wide")
st.title("Release Incident Advisor Chatbot")
st.caption("Ask things like: 'What incidents occurred last quarter?', 'What happened on 2025-03-24', 'What did the release team work on in June?' ")
st.caption("This is a prototype for internal incident search using FAISS and OpenAI's API.")
st.caption("Practice project by: Elton Zhang.")

query = st.text_input("Your Question", placeholder="e.g., Any QA issues this month?")

# === Helper: Convert Relative Time Phrase to Date Range ===
def resolve_relative_date(query):
    today = datetime.today()
    year = today.year
    month = today.month
    quarter = (month - 1) // 3 + 1

    query_lower = query.lower()
    if "this year" in query_lower:
        return f"{year}-01-01", today.strftime("%Y-%m-%d")
    if "last year" in query_lower:
        return f"{year-1}-01-01", f"{year-1}-12-31"
    if "this quarter" in query_lower:
        start_month = 3 * (quarter - 1) + 1
        start_date = datetime(year, start_month, 1)
        return start_date.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
    if "last quarter" in query_lower:
        if quarter == 1:
            return f"{year-1}-10-01", f"{year-1}-12-31"
        start_month = 3 * (quarter - 2) + 1
        start_date = datetime(year, start_month, 1)
        end_date = datetime(year, start_month + 2, 28)
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    if "this month" in query_lower:
        return f"{year}-{month:02d}-01", today.strftime("%Y-%m-%d")
    if "last month" in query_lower:
        last_month = month - 1 if month > 1 else 12
        last_year = year if month > 1 else year - 1
        return f"{last_year}-{last_month:02d}-01", f"{last_year}-{last_month:02d}-31"
    return None, None

# === Begin Filtering ===
filtered_docs = documents

# 1. Issue Owner Filtering
owner_aliases = {
    "release": "Release Engineering",
    "release engineering": "Release Engineering",
    "qa": "QA",
    "ssm": "SSM",
    "dba": "DBA",
    "scm": "SCM",
    "network": "Network",
    "middleware": "Middleware",
    "prod ops": "Prod Ops",
    "data services": "Data Services"
}
owner_match = None
for key in owner_aliases:
    if key in query.lower():
        owner_match = owner_aliases[key]
        break
if owner_match:
    st.info(f"🎯 Filtering for incidents owned by **{owner_match}**")
    filtered_docs = [doc for doc in filtered_docs if owner_match in doc]

# 2. Exact Date Filter (YYYY-MM-DD)
exact_date_match = re.search(r'\b(2025-\d{2}-\d{2})\b', query)
if exact_date_match:
    date_str = exact_date_match.group(1)
    st.info(f"📅 Filtering for incidents on **{date_str}**")
    filtered_docs = [doc for doc in filtered_docs if date_str in doc]

# 3. Month Name Filter
month_match = re.search(
    r'in\s+(january|february|march|april|may|june|july|august|september|october|november|december)',
    query.lower()
)
if month_match:
    month_str = month_match.group(1)
    month_number = datetime.strptime(month_str, "%B").month
    st.info(f"📆 Filtering incidents for **{month_str.title()}**")
    filtered_docs = [doc for doc in filtered_docs if f"-{month_number:02d}-" in doc]

# 4. Relative Date Filter
start_date, end_date = resolve_relative_date(query)
if start_date and end_date:
    st.info(f"⏱️ Filtering incidents from **{start_date}** to **{end_date}**")
    filtered_docs = [
        doc for doc in filtered_docs
        if start_date <= doc[:10] <= end_date  # assumes date is at start of string
    ]

# === Retrieval & LLM ===
if query:
    if len(filtered_docs) == 0:
        st.warning("No incidents found matching that filter.")
    else:
        filtered_embeddings = embedder.encode(filtered_docs)
        temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
        temp_index.add(filtered_embeddings)

        q_embedding = embedder.encode([query])
        _, I = temp_index.search(q_embedding, k=min(5, len(filtered_docs)))
        top_docs = [filtered_docs[i] for i in I[0]]

        with st.expander("🔍 Retrieved Incident Records"):
            for i, doc in enumerate(top_docs):
                st.markdown(f"**Incident {i+1}:** {doc}")

        # Build prompt
        context = "\n\n".join(top_docs)
        prompt = f"""
You are an assistant analyzing internal production incident reports.

User Question:
{query}

Relevant Incident Records:
{context}

Based on these incidents, answer the user's question. If no direct answer exists, summarize what can be inferred.
"""

        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=500,
            )
            st.markdown("### 🤖 Answer")
            st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"⚠️ OpenAI API Error: {e}")
