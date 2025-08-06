
import streamlit as st
from openai import OpenAI
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
import pandas as pd
import re
import calendar

st.set_page_config(page_title="Release Incident Advisor Chatbot", layout="wide")
st.title("Release Incident Advisor Chatbot")
st.caption("Ask things like: 'What incidents occurred last quarter?', 'What happened on 2025-03-24', 'What did the release team work on in June?' ")
st.caption("This is a prototype for internal incident search using FAISS and OpenAI's API.")
st.caption("Practice project by: Elton Zhang.")

with open("incident_texts.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

index = faiss.read_index("incident_index.faiss")
model = SentenceTransformer("all-MiniLM-L6-v2")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "token_count" not in st.session_state:
    st.session_state.token_count = 0

st.sidebar.title("Usage Stats")
st.sidebar.metric("Queries this session", st.session_state.query_count)
st.sidebar.metric("Total tokens used", st.session_state.token_count)
estimated_cost = st.session_state.token_count * 0.000002
st.sidebar.metric("Estimated cost", f"${estimated_cost:.4f}")

def extract_metadata(entry):
    try:
        date_str, rest = entry.split(" - ", 1)
        date = datetime.strptime(date_str.strip(), "%Y-%m-%d")
        team = rest.split(":")[0].strip()
        root_match = re.search(r"Root Cause: (.*?)\s*\|", entry)
        root_cause = root_match.group(1).strip() if root_match else "Unknown"
        return {"date": date, "month": date.strftime("%B"), "year": date.year, "team": team, "root_cause": root_cause, "entry": entry}
    except:
        return {"date": None, "month": "Unknown", "year": "Unknown", "team": "Unknown", "root_cause": "Unknown", "entry": entry}

incident_df = pd.DataFrame([extract_metadata(doc) for doc in documents if doc])

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

with st.expander("📁 Incident Explorer", expanded=False):
    team_filter = st.selectbox("Filter by Team", options=["All"] + sorted(incident_df["team"].unique()))
    filtered_df = incident_df if team_filter == "All" else incident_df[incident_df["team"] == team_filter]
    st.dataframe(filtered_df[["date", "team", "root_cause"]].sort_values("date", ascending=False))

query = st.text_input("Your Question", placeholder="e.g., Any QA issues this month?")

def respond_with_date_match(date_str):
    matched = incident_df[incident_df["date"] == date_str]
    if not matched.empty:
        st.subheader("📅 Incident on " + date_str.strftime("%Y-%m-%d"))
        for row in matched["entry"]:
            st.text(row)
    else:
        st.write(f"No incidents recorded on {date_str.strftime('%Y-%m-%d')}.")

def respond_with_month_count(month_name):
    count = incident_df[incident_df["month"] == month_name].shape[0]
    st.write(f"There were **{count}** incidents reported in **{month_name}**.")

def filter_by_date_range(start, end, label):
    filtered = incident_df[(incident_df["date"] >= start) & (incident_df["date"] <= end)]
    st.subheader(f"📅 Incidents during {label}")
    if filtered.empty:
        st.write(f"No incidents recorded during {label}.")
    else:
        for row in filtered["entry"]:
            st.text(row)

if query:
    st.session_state.query_count += 1
    query_lower = query.lower()

    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", query)
    month_match = re.search(r"(\b" + "\b|\b".join(calendar.month_name[1:]) + "\b)", query, re.IGNORECASE)
    how_many = "how many" in query_lower

    today = datetime.today()
    this_month_start = today.replace(day=1)
    last_month_end = this_month_start - timedelta(days=1)
    last_month_start = last_month_end.replace(day=1)
    current_year = today.year
    last_year = current_year - 1

    current_quarter = (today.month - 1) // 3 + 1
    this_q_start = datetime(today.year, 3 * current_quarter - 2, 1)
    last_q_end = this_q_start - timedelta(days=1)
    last_q_start = datetime(last_q_end.year, 3 * ((last_q_end.month - 1)//3) + 1, 1)

    if date_match:
        try:
            date_obj = datetime.strptime(date_match.group(1), "%Y-%m-%d")
            respond_with_date_match(date_obj)
        except ValueError:
            st.warning("Invalid date format detected.")
    elif how_many and month_match:
        respond_with_month_count(month_match.group(1).capitalize())
    elif "this year" in query_lower:
        filter_by_date_range(datetime(current_year, 1, 1), datetime(current_year, 12, 31), f"{current_year}")
    elif "last year" in query_lower:
        filter_by_date_range(datetime(last_year, 1, 1), datetime(last_year, 12, 31), f"{last_year}")
    elif "this month" in query_lower:
        filter_by_date_range(this_month_start, today, "this month")
    elif "last month" in query_lower:
        filter_by_date_range(last_month_start, last_month_end, "last month")
    elif "this quarter" in query_lower:
        filter_by_date_range(this_q_start, today, "this quarter")
    elif "last quarter" in query_lower:
        filter_by_date_range(last_q_start, last_q_end, "last quarter")
    else:
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
