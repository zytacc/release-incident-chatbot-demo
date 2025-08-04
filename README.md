# 🛠️ Release Incident Chatbot

A lightweight AI-powered assistant that helps engineering teams search and summarize historical production incident records using natural language.
Built by Elton Zhang as a personal learning project to explore applied AI in incident management and DevOps tooling.

🔗 **Live Demo**: [https://release-incident-chatbot-demo-8ydtmw2zvcffgfcmyqdfcw.streamlit.app/](https://release-incident-chatbot-demo-8ydtmw2zvcffgfcmyqdfcw.streamlit.app/)

---

## 🧠 What It Does

This chatbot allows you to:
- Query past production release incidents using **natural language**
- Filter by **date**, **month**, **issue owner**, or **relative timeframes**
- Summarize matching incidents using **OpenAI's GPT-3.5 Turbo**
- Surface root causes and resolution patterns across deployments

Example queries:
- `"What incidents occurred in June?"`
- `"What did the release team handle last quarter?"`
- `"How many issues happened on 2025-03-24?"`
- `"Any QA incidents this month?"`

---

## 🧰 Tech Stack

| Layer            | Tools Used                                  |
|------------------|---------------------------------------------|
| UI               | Streamlit                                   |
| Semantic Search  | FAISS + Sentence Transformers (MiniLM)      |
| LLM Integration  | OpenAI GPT-3.5 Turbo (via API)              |
| Data Format      | JSON log corpus of mock incident summaries  |
| Deployment       | Streamlit Cloud                             |

---

## 🚀 How It Works

1. Incidents are embedded using `sentence-transformers` and stored in a FAISS vector index.
2. User queries are semantically matched against the index.
3. Matching incidents are passed to OpenAI’s chat model for summarization or analysis.
4. Responses are shown alongside the retrieved logs for transparency.


