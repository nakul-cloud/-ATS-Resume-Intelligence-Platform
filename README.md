# 🧠 ATS Resume Intelligence Platform (Agentic + AI Interview Evaluation)

An end-to-end **AI-powered Applicant Tracking + Career Intelligence System** that
parses resumes, stores semantic embeddings, matches candidates to job descriptions,
and evaluates interview readiness using **LLM Agents + LangGraph**.

Built with **FastAPI**, **Groq + Gemini**, **Supabase (pgvector)**, and **LangGraph**.

---

## 🚀 Key Features

### 📄 1. Resume Upload & Parsing
- Upload PDF resume
- Extracts structured fields:
  - Name, Email, Phone
  - Primary Role & Domain
  - Experience & Education
  - Skills (JSON structured)
- Powered by **Groq + Gemini LLM**
- Retry-safe JSON enforcement (LangGraph)

---

### 🧠 2. Semantic Embeddings (pgvector)
- Resume text → Embeddings (768-dim)
- Stored in Supabase PostgreSQL
- Query via pgvector similarity search
- Used for candidate–JD matching

---

### 🎯 3. Job Description (JD) Matching
- Paste JD → auto-embeds & compares
- Matches against **66+ stored resumes**
- Optional domain-based filtering
- Outputs:
  - Similarity score
  - Strengths & gaps
  - Interview question suggestions

---

### 🧩 4. Agentic Self Evaluation (New)
Upload your **own resume + JD** → Runs a multi-agent evaluation:

| Agent | Purpose |
|-------|----------|
| Self Evaluation Agent | Extract, parse, match, score |
| Decision Agent | Classifies: **Low / Medium / High** readiness |
| Confidence Agent | (<50) Fix fundamentals + build project plan |
| Gap Analysis Agent | (50-79) Targeted improvement roadmap |
| Interview Prep Agent | (≥80) Advanced role-focused prep |

🧾 Output:
- Fit Score out of 100
- Strengths & Skill Gaps
- Personalized Learning Roadmap
- Confidence feedback
- Practice interview questions *(score ≥ 60 only)*

---

### 🎙️ 5. Interactive Interview Evaluation Agent (New)
Simulates a **live interview feedback loop**:

User answers → Agent evaluates on:
✔ Accuracy, clarity, relevance  
✔ Strengths & weaknesses  
✔ Score (0-100)  
✔ Follow-up question  
✔ Difficulty adjusts dynamically

📌 *Low scorers (<60) are NOT given questions — only improvement feedback.*

---

## 🧱 System Architecture

