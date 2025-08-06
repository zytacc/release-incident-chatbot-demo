
import streamlit as st
from openai import OpenAI
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
import pandas as pd
import re

# --- Streamlit Page Config ---
st.set_page_config(page_title="Release Incident Advisor Chatbot", layout="wide")
st.title("Release Incident Advisor Chatbot")
st.caption("Ask things like: 'What incidents occurred last quarter?', 'What happened on 2025-03-24', 'What did the release team work on in June?' ")
st.caption("This is a prototype for internal incident search using FAISS and OpenAI's API.")
st.caption("Practice project by: Elton Zhang.")

# --- Load documents and FAISS index ---
with open("incident_texts.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

index = faiss.read_index("incident_index.faiss")
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Setup OpenAI client ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Session state for usage tracking ---
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "token_count" not in st.session_state:
    st.session_state.token_count = 0

# --- Sidebar usage metrics ---
st.sidebar.title("Usage Stats")
st.sidebar.metric("Queries this session", st.session_state.query_count)
st.sidebar.metric("Total tokens used", st.session_state.token_count)
estimated_cost = st.session_state.token_count * 0.000002
st.sidebar.metric("Estimated cost", f"${estimated_cost:.4f}")

# --- Incident metadata extraction ---
def extract_metadata(entry):
    try:
        date_str, rest = entry.split(" - ", 1)
        date = datetime.strptime(date_str.strip(), "%Y-%m-%d")
        team = rest.split(":")[0].strip()
        root_match = re.search(r"Root Cause: (.*?)\s*\|", entry)
        root_cause = root_match.group(1).strip() if root_match else "Unknown"
        return {"date": date, "month": date.strftime("%B"), "team": team, "root_cause": root_cause, "entry": entry}
    except:
        return {"date": None, "month": "Unknown", "team": "Unknown", "root_cause": "Unknown", "entry": entry}

incident_df = pd.DataFrame([extract_metadata(doc) for doc in documents if doc])

# --- Dashboard section ---
with st.expander("📊 Incident Dashboard", expanded=False):
    st.subheader("Incidents by Month")
    month_df = incident_df["month"].value_counts().sort_index()
    st.bar_chart(month_df)

    st.subheader("Incidents by Team")
    team_df = incident_df["team"].value_counts().sort_values(ascending=False)
    st.dataframe(team_df.rename("Count").reset_index().rename(columns={"index": "Team"}))

    st.subheader("Top Root Causes")
    root_df = incident_df["root_cause"].value_counts().head(10)
    st.dataframe(root_df.rename("Count").reset_index().rename(columns={"index": "Root Cause"}))

# --- Explorer section ---
with st.expander("📁 Incident Explorer", expanded=False):
    team_filter = st.selectbox("Filter by Team", options=["All"] + sorted(incident_df["team"].unique()))
    if team_filter != "All":
        filtered_df = incident_df[incident_df["team"] == team_filter]
    else:
        filtered_df = incident_df
    st.dataframe(filtered_df[["date", "team", "root_cause"]].sort_values("date", ascending=False))

# --- Query input ---
query = st.text_input("Your Question", placeholder="e.g., Any QA issues this month?")

if query:
    st.session_state.query_count += 1
    query_embedding = model.encode([query])
    top_k = 5
    _, I = index.search(np.array(query_embedding), top_k)
    matched_docs = [documents[i] for i in I[0]]
    context = "\n".join(matched_docs)

    messages = [
        {"role": "system", "content": "You are an assistant that answers based on historical incident records."},
        {"role": "user", "content": f"Relevant incidents:\n{context}\n\nQuestion: {query}"}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        answer = response.choices[0].message.content
        usage = response.usage
        st.session_state.token_count += usage.total_tokens

        st.subheader("🔍 Retrieved Incident Records")
        for doc in matched_docs:
            st.text(doc)

        st.subheader("🤖 Answer")
        st.write(answer)
        st.caption(f"Tokens used: {usage.total_tokens} (Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens})")

    except Exception as e:
        st.error(f"⚠️ OpenAI API Error: {e}")
