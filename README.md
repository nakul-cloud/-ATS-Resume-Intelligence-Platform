# 🧠 ATS Resume Intelligence Platform

An **AI-powered Applicant Tracking System (ATS)** that parses resumes, stores semantic embeddings, and intelligently matches candidates to job descriptions using **LLMs + pgvector + RAG-style retrieval**.

Built with **FastAPI, Gemini (Google Generative AI), Supabase (PostgreSQL + pgvector), and LangGraph**.

---

## 🚀 Key Features

### 📄 Resume Upload & Parsing
- Upload PDF resumes
- Extracts:
  - Candidate name
  - Email & phone
  - Primary role & domain
  - Experience & education
  - Skills (structured)
- Uses **Gemini LLM** for accurate parsing
- Stores results in **Supabase (PostgreSQL)**

---

### 🧠 Semantic Embeddings (pgvector)
- Resume content converted into **768-dim embeddings**
- Generated using **Gemini text-embedding-004**
- Stored in PostgreSQL via **pgvector**
- Enables **semantic similarity search**

---

### 🎯 Job Description Matching
- Paste a Job Description
- JD → embedding → similarity search
- Retrieves **Top-K most relevant candidates**
- Optional **hard domain filtering**
- Results ranked by **vector similarity**
- Each candidate scored using **Gemini evaluation**

---

### 🔁 Resilient Resume Parsing (LangGraph)
- Uses **LangGraph workflow**
- Retry-safe parsing if JSON fails
- Structured state transitions:
  - PDF extraction
  - LLM parsing
  - Embedding generation
  - Database storage

---

### 🔍 Self Evaluation
- Upload your own resume + JD
- Get:
  - Fit score (0–100)
  - Strengths
  - Skill gaps
  - Interview questions
- **No data stored** (privacy-safe)

---



---

## 🧰 Tech Stack

### Backend
- **FastAPI** (Python)
- **LangGraph** – resilient LLM workflows
- **Google Gemini**
  - `gemini-2.5-flash-lite` (parsing & evaluation)
  - `text-embedding-004` (embeddings)

### Database
- **Supabase**
  - PostgreSQL
  - pgvector extension

### Frontend
- HTML + Tailwind CSS
- Vanilla JavaScript
- Responsive ATS-style UI

---

## 📦 Database Schema (Core Tables)

- `candidates_parsed`
  - Candidate details
  - Resume embedding (pgvector)
- `candidate_skills`
  - Normalized skills
- `profiles`
  - Extended profile text + embedding
- `resumes_raw`
  - Uploaded files metadata

---

 

