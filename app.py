import os
import json
from io import BytesIO
from typing import List, Optional, Any, Dict, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel
from typing_extensions import TypedDict
from collections import Counter

import pdfplumber
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END, START
# -------------------------

import google.generativeai as genai
from supabase import create_client, Client
# --- Groq Imports ---
from groq import Groq

# =====================================================
#                       CONFIG
# =====================================================

GEMINI_API_KEY = 
GROQ_API_KEY = 
SUPABASE_URL = 
SUPABASE_KEY = "

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment variables")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in environment variables")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase credentials not set in environment variables")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Use Groq models for text generation and parsing
resume_model_name = "openai/gpt-oss-120b"  # Groq model for resume parsing
evaluate_model_name = "openai/gpt-oss-120b"  # Groq model for evaluation

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# =====================================================
#                       FASTAPI APP
# =====================================================

app = FastAPI(title="ATS Resume Intelligence Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
#                       MODELS
# =====================================================

class EvaluateJDRequest(BaseModel):
    jd_text: str
    domain: Optional[str] = None
    top_k: int = 5

class CandidateEval(BaseModel):
    candidate_id: int
    candidate_name: Optional[str]
    primary_role: Optional[str]
    primary_domain: Optional[str]
    total_experience: Optional[float]
    score_100: float
    strengths: List[str]
    gaps: List[str]
    interview_questions: List[str]

class EvaluateJDResponse(BaseModel):
    jd_text: str
    domain_filter: Optional[str]
    results: List[CandidateEval]

class SelfEvalResponse(BaseModel):
    score_100: float
    strengths: List[str]
    gaps: List[str]
    interview_questions: List[str]
    role: Optional[str]
    domain: Optional[str]

# NEW: Agentic Self Evaluation Response
class AgentSelfEvalResponse(BaseModel):
    score_100: float
    strengths: List[str]
    gaps: List[str]
    interview_questions: List[str]
    learning_roadmap: List[str]
    confidence_feedback: str
    decision_reasoning: str
    status: str  # success | partial | failed

# NEW: Interview Evaluation Models
class InterviewAnswerRequest(BaseModel):
    session_id: str
    question: str
    user_answer: str
    role: Optional[str]
    domain: Optional[str]

class InterviewEvalResponse(BaseModel):
    answer_score: int
    feedback: str
    strengths: List[str]
    weaknesses: List[str]
    follow_up_question: str
    next_difficulty: str
    status: str   # success | failed

class MetricsResponse(BaseModel):
    key_metrics: Dict[str, Any]
    score_distribution: Dict[str, Any]
    top_skills: Dict[str, Any]
    domain_distribution: Dict[str, Any]
    monthly_activity: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    database_stats: Dict[str, Any]
    recent_uploads: List[Dict[str, Any]]
    top_domains: List[Dict[str, Any]]

# =====================================================
#                   HELPER FUNCTIONS
# =====================================================

def extract_pdf_text(pdf_bytes: bytes, max_chars: int = 6000) -> str:
    """Extract text from PDF"""
    text_chunks = []
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            text_chunks.append(text)
    return "\n".join(text_chunks)[:max_chars]

def parse_resume_simple(text: str, prompt_modifier: str = "") -> dict:
    prompt = f"""
You are an ATS resume parser. {prompt_modifier}
Return ONLY valid JSON in this exact structure:
{{
  "candidate_name": "",
  "email": "",
  "phone_number": "",
  "primary_role_title": "",
  "primary_domain": "",
  "total_experience_years": 0,
  "highest_education": "",
  "summary_text": "",
  "skills": [{{ "skill_name": "" }}]
}}
"""

    # Use Groq API instead of Gemini
    try:
        response = groq_client.chat.completions.create(
            model=resume_model_name,
            messages=[
                {"role": "system", "content": "You are an ATS resume parser."},
                {"role": "user", "content": f"{prompt}\n\nResume Text:\n{text}"}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        raw = response.choices[0].message.content
        
    except Exception as e:
        raise RuntimeError(f"Groq API call failed: {str(e)}")

    if not raw:
        raise RuntimeError("Groq returned empty response")

    # Clean the response
    raw = raw.replace("```json", "").replace("```", "").strip()

    print("🔍 GROQ RAW OUTPUT:\n", raw)

    try:
        return json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Groq JSON parse failed: {raw}") from e

def generate_candidate_text_for_embedding(parsed: dict) -> str:
    """Generate text for embedding"""
    skills_list = [s.get("skill_name", "") for s in parsed.get("skills", []) if s.get("skill_name")]
    skills_text = ", ".join(skills_list) if skills_list else "N/A"
    
    return f"""
Name: {parsed.get('candidate_name') or 'N/A'}
Role: {parsed.get('primary_role_title') or 'N/A'}
Domain: {parsed.get('primary_domain') or 'N/A'}
Experience: {parsed.get('total_experience_years') or 'N/A'} years

Summary:
{parsed.get('summary_text') or 'N/A'}

Skills:
{skills_text}
""".strip()

def embed_text(text: str) -> List[float]:
    """Generate embedding using Gemini (keeping this as is)"""
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="RETRIEVAL_DOCUMENT",
    )
    return response["embedding"]

def evaluate_candidate_with_groq(jd_text: str, candidate_text: str) -> dict:
    """Evaluate candidate using Groq API"""
    prompt = """
You are an expert technical recruiter. Evaluate how well this candidate fits the job description.
Return ONLY valid JSON:
{
  "score_100": 0-100,
  "strengths": [],
  "gaps": [],
  "interview_questions": []
}
"""
    
    try:
        response = groq_client.chat.completions.create(
            model=evaluate_model_name,
            messages=[
                {"role": "system", "content": "You are an expert technical recruiter."},
                {"role": "user", "content": f"{prompt}\n\nJOB DESCRIPTION:\n{jd_text}\n\nCANDIDATE:\n{candidate_text}"}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        raw = response.choices[0].message.content
        
    except Exception as e:
        raise RuntimeError(f"Groq evaluation API call failed: {str(e)}")

    if not raw:
        raise RuntimeError("Groq returned empty response for evaluation")

    raw = raw.replace("```json", "").replace("```", "").strip()
    
    try:
        return json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Groq evaluation JSON parse failed: {raw}") from e

# NEW: Interview Evaluation Tool
def evaluate_interview_answer_with_groq(
    question: str,
    user_answer: str,
    role: str = "",
    domain: str = ""
) -> Dict:
    """Evaluate interview answer using Groq API"""
    prompt = """
You are an expert technical interviewer. Evaluate the user's answer to an interview question.
Return ONLY valid JSON in this exact structure:
{
  "answer_score": 0-100,
  "strengths": [],
  "weaknesses": [],
  "feedback": "",
  "follow_up_question": "",
  "next_difficulty": "easier | same | harder"
}

Evaluation guidelines:
- Score based on technical accuracy, clarity, and relevance
- strengths: what the user did well
- weaknesses: areas for improvement
- feedback: constructive feedback for the user
- follow_up_question: a relevant follow-up question
- next_difficulty: adjust based on score (≤50: easier, 51-80: same, ≥81: harder)
"""
    
    try:
        # Build context based on role and domain
        context = f"Role: {role}\nDomain: {domain}\n" if role or domain else ""
        
        response = groq_client.chat.completions.create(
            model=evaluate_model_name,
            messages=[
                {"role": "system", "content": "You are an expert technical interviewer."},
                {"role": "user", "content": f"{prompt}\n\n{context}Question: {question}\n\nUser Answer: {user_answer}"}
            ],
            temperature=0.3,  # Lower temperature for consistent evaluation
            max_tokens=1500
        )
        
        raw = response.choices[0].message.content
        
    except Exception as e:
        raise RuntimeError(f"Groq interview evaluation API call failed: {str(e)}")

    if not raw:
        raise RuntimeError("Groq returned empty response for interview evaluation")

    # Clean the response
    raw = raw.replace("```json", "").replace("```", "").strip()
    
    try:
        return json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Groq interview evaluation JSON parse failed: {raw}") from e

# =====================================================
#                   DATABASE OPERATIONS
# =====================================================

def store_resume_in_supabase(file_name: str, parsed: dict, embedding: List[float]) -> Dict:
    try:
        skills_list = [
            s.get("skill_name") for s in parsed.get("skills", [])
            if s.get("skill_name")
        ]
        skills_text = ", ".join(skills_list) if skills_list else None

        data = {
            "candidate_name": parsed.get("candidate_name"),
            "email": parsed.get("email"),
            "phone_number": parsed.get("phone_number"),
            "primary_role_title": parsed.get("primary_role_title"),
            "primary_domain": parsed.get("primary_domain"),
            "total_experience_years": parsed.get("total_experience_years"),
            "highest_education": parsed.get("highest_education"),
            "summary_text": parsed.get("summary_text"),
            "skills_text": skills_text,
            "embedding": embedding,
            "created_at": datetime.utcnow().isoformat()
        }

        res = supabase.table("candidates_parsed").insert(data).execute()

        if not res.data:
            raise RuntimeError(f"Supabase insert failed: {res}")

        candidate_id = res.data[0]["id"]

        if skills_list:
            rows = [{"candidate_id": candidate_id, "skill_name": s} for s in skills_list]
            supabase.table("candidate_skills").insert(rows).execute()

        return {
            "candidate_id": candidate_id,
            "status": "stored",
            "embedding_dimensions": len(embedding)
        }

    except Exception as e:
        raise Exception(f"Failed to store in Supabase: {str(e)}")

def search_candidates_by_similarity(
    query_embedding: List[float],
    top_k: int = 5,
    domain: Optional[str] = None
) -> List[Dict]:
    """Search candidates using pgvector similarity."""
    try:
        payload = {
            "query_embedding": query_embedding,
            "match_count": top_k,
            "domain_filter": domain
        }

        result = supabase.rpc("match_candidates", payload).execute()

        if result.data:
            return result.data

        print("pgvector search returned no results, using fallback")
        return search_candidates_fallback(top_k=top_k)

    except Exception as e:
        print("pgvector search failed:", e)
        return search_candidates_fallback(top_k=top_k)

def search_candidates_fallback(top_k: int = 5) -> List[Dict]:
    """Safe fallback when pgvector search fails."""
    try:
        result = (
            supabase
            .table("candidates_parsed")
            .select(
                "id, candidate_name, primary_role_title, primary_domain, "
                "total_experience_years, summary_text, skills_text"
            )
            .order("created_at", desc=True)
            .limit(top_k)
            .execute()
        )

        candidates = result.data or []

        for c in candidates:
            c["similarity"] = 0.5

        return candidates

    except Exception as e:
        print("Fallback search error:", e)
        return []

# =====================================================
#                   METRICS FUNCTIONS
# =====================================================

def get_database_metrics() -> Dict[str, Any]:
    """Get comprehensive database metrics"""
    try:
        # Get total candidates
        candidates_result = supabase.table("candidates_parsed").select("count", count="exact").execute()
        total_candidates = candidates_result.count or 0
        
        # Get recent uploads (last 7 days)
        week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
        recent_result = supabase.table("candidates_parsed")\
            .select("created_at")\
            .gte("created_at", week_ago)\
            .execute()
        recent_uploads = len(recent_result.data) if recent_result.data else 0
        
        # Get skills count
        skills_result = supabase.table("candidate_skills").select("skill_name").execute()
        skills_list = [skill["skill_name"] for skill in skills_result.data] if skills_result.data else []
        unique_skills = len(set(skills_list))
        
        # Get domain distribution
        domain_result = supabase.table("candidates_parsed")\
            .select("primary_domain")\
            .execute()
        
        domains = [d["primary_domain"] for d in domain_result.data if d.get("primary_domain")]
        domain_counts = Counter(domains)
        
        # Get experience statistics
        experience_result = supabase.table("candidates_parsed")\
            .select("total_experience_years")\
            .execute()
        
        experiences = [e["total_experience_years"] for e in experience_result.data if e.get("total_experience_years")]
        avg_experience = round(np.mean(experiences), 1) if experiences else 0
        
        # Calculate monthly activity (last 6 months)
        monthly_activity = []
        for i in range(6):
            month_start = (datetime.utcnow().replace(day=1) - timedelta(days=30*i)).replace(day=1)
            month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            
            month_result = supabase.table("candidates_parsed")\
                .select("count", count="exact")\
                .gte("created_at", month_start.isoformat())\
                .lt("created_at", month_end.isoformat())\
                .execute()
            
            monthly_activity.append({
                "month": month_start.strftime("%b"),
                "uploads": month_result.count or 0
            })
        
        monthly_activity.reverse()
        
        return {
            "total_candidates": total_candidates,
            "recent_uploads_7d": recent_uploads,
            "unique_skills": unique_skills,
            "domain_distribution": dict(domain_counts.most_common(10)),
            "avg_experience_years": avg_experience,
            "monthly_activity": monthly_activity
        }
        
    except Exception as e:
        print(f"Error getting database metrics: {e}")
        return {}

def calculate_score_distribution() -> Dict[str, Any]:
    """Calculate match score distribution from recent evaluations"""
    try:
        # For now, simulate score distribution
        # In production, you would query a table that stores evaluation results
        return {
            "labels": ["0-50%", "51-60%", "61-70%", "71-80%", "81-90%", "91-100%"],
            "data": [45, 78, 120, 245, 189, 92]
        }
    except Exception as e:
        print(f"Error calculating score distribution: {e}")
        return {"labels": [], "data": []}

def get_top_skills(limit: int = 10) -> Dict[str, Any]:
    """Get top skills from database"""
    try:
        skills_result = supabase.table("candidate_skills").select("skill_name").execute()
        
        if not skills_result.data:
            return {"labels": [], "data": []}
        
        skills_list = [skill["skill_name"] for skill in skills_result.data]
        skill_counts = Counter(skills_list)
        
        top_skills = skill_counts.most_common(limit)
        
        return {
            "labels": [skill[0] for skill in top_skills],
            "data": [skill[1] for skill in top_skills]
        }
        
    except Exception as e:
        print(f"Error getting top skills: {e}")
        return {"labels": [], "data": []}

def get_monthly_activity_data() -> Dict[str, Any]:
    """Get monthly activity data for charts"""
    try:
        # Get last 7 months of data
        labels = []
        uploads = []
        matches = []  # Simulated for now
        
        for i in range(6, -1, -1):
            month = datetime.utcnow() - timedelta(days=30*i)
            labels.append(month.strftime("%b"))
            
            # Get uploads for this month
            month_start = month.replace(day=1)
            month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            
            upload_result = supabase.table("candidates_parsed")\
                .select("count", count="exact")\
                .gte("created_at", month_start.isoformat())\
                .lt("created_at", month_end.isoformat())\
                .execute()
            
            uploads.append(upload_result.count or 0)
            
            # Simulate matches (would come from evaluations table in production)
            matches.append(int((upload_result.count or 0) * 0.8))
        
        return {
            "labels": labels,
            "datasets": [
                {"label": "Resume Uploads", "data": uploads},
                {"label": "Job Matches", "data": matches}
            ]
        }
        
    except Exception as e:
        print(f"Error getting monthly activity: {e}")
        return {"labels": [], "datasets": []}

def get_performance_metrics() -> Dict[str, Any]:
    """Calculate system performance metrics"""
    try:
        # Simulate performance metrics for now
        # In production, these would be calculated from actual performance data
        return {
            "parsing_accuracy": 94.2,
            "embedding_quality": 91.5,
            "match_relevance": 86.7,
            "response_time": 2.4,
            "uptime_percentage": 99.8,
            "error_rate": 0.3
        }
        
    except Exception as e:
        print(f"Error calculating performance metrics: {e}")
        return {}

def get_recent_uploads(limit: int = 5) -> List[Dict[str, Any]]:
    """Get recent resume uploads"""
    try:
        result = supabase.table("candidates_parsed")\
            .select("id, candidate_name, primary_role_title, primary_domain, created_at")\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()
        
        if not result.data:
            return []
        
        recent_uploads = []
        for candidate in result.data:
            recent_uploads.append({
                "id": candidate["id"],
                "name": candidate.get("candidate_name", "Unknown"),
                "role": candidate.get("primary_role_title", "N/A"),
                "domain": candidate.get("primary_domain", "N/A"),
                "upload_date": candidate.get("created_at", "").split("T")[0] if candidate.get("created_at") else "N/A"
            })
        
        return recent_uploads
        
    except Exception as e:
        print(f"Error getting recent uploads: {e}")
        return []

# =====================================================
#          INTERVIEW EVALUATION: STATE & AGENTS
# =====================================================

class InterviewEvalState(TypedDict):
    """State for the interview evaluation workflow."""
    # Input
    question: str
    user_answer: str
    role: str
    domain: str
    
    # Output
    answer_score: int
    feedback: str
    strengths: List[str]
    weaknesses: List[str]
    follow_up_question: str
    next_difficulty: str
    error: str

def interview_evaluation_agent(state: InterviewEvalState) -> InterviewEvalState:
    """Node: Evaluates interview answers using LLM tool."""
    print("--- INTERVIEW AGENT: Evaluating Answer ---")
    
    try:
        # Call the LLM tool for evaluation
        evaluation = evaluate_interview_answer_with_groq(
            question=state["question"],
            user_answer=state["user_answer"],
            role=state.get("role", ""),
            domain=state.get("domain", "")
        )
        
        # Store all returned fields into state
        return {
            **state,
            "answer_score": evaluation.get("answer_score", 0),
            "feedback": evaluation.get("feedback", ""),
            "strengths": evaluation.get("strengths", []),
            "weaknesses": evaluation.get("weaknesses", []),
            "follow_up_question": evaluation.get("follow_up_question", ""),
            "next_difficulty": evaluation.get("next_difficulty", "same"),
            "error": ""  # Clear error on success
        }
        
    except Exception as e:
        print(f"Interview evaluation agent error: {str(e)}")
        # Provide safe fallback values
        return {
            **state,
            "answer_score": 0,
            "feedback": "Unable to evaluate answer due to system error. Please try again.",
            "strengths": [],
            "weaknesses": ["System error occurred during evaluation"],
            "follow_up_question": "Can you explain this in another way?",
            "next_difficulty": "same",
            "error": f"Evaluation failed: {str(e)[:100]}"
        }

# =====================================================
#          INTERVIEW EVALUATION: WORKFLOW
# =====================================================

# Create the interview evaluation workflow
interview_eval_workflow = StateGraph(InterviewEvalState)

# Add nodes
interview_eval_workflow.add_node("interview_eval_agent", interview_evaluation_agent)

# Set entry point and edges
interview_eval_workflow.set_entry_point("interview_eval_agent")
interview_eval_workflow.add_edge("interview_eval_agent", END)

# Compile the workflow
interview_eval_app = interview_eval_workflow.compile()

# =====================================================
#          AGENTIC SELF-EVALUATION: STATE & AGENTS
# =====================================================

class SelfEvalState(TypedDict):
    """State for the agentic self-evaluation workflow."""
    # Input
    pdf_bytes: bytes
    jd_text: str
    resume_text: str
    
    # Core Evaluation Results
    score_100: float
    strengths: List[str]
    gaps: List[str]
    interview_questions: List[str]
    role: str
    domain: str
    
    # Agentic Decisions & Outputs
    learning_roadmap: List[str]
    confidence_feedback: str
    next_action: str
    decision_reasoning: str
    error: str
    current_step: str
    
    # Internal
    parsed_resume: Dict[str, Any]
    candidate_text: str

def self_eval_agent(state: SelfEvalState) -> SelfEvalState:
    """Node 1: Uses existing tools to parse and evaluate the resume."""
    print("--- AGENT 1: Self Evaluation Agent ---")
    
    try:
        # Step 1: Extract text from PDF
        resume_text = extract_pdf_text(state["pdf_bytes"])
        
        # Step 2: Parse resume using existing tool
        parsed_resume = parse_resume_simple(resume_text)
        
        # Step 3: Generate candidate text for evaluation
        candidate_text = generate_candidate_text_for_embedding(parsed_resume)
        
        # Step 4: Evaluate against JD using existing tool
        evaluation = evaluate_candidate_with_groq(state["jd_text"], candidate_text)
        
        # Store results in state
        update = {
            "resume_text": resume_text,
            "parsed_resume": parsed_resume,
            "candidate_text": candidate_text,
            "score_100": evaluation["score_100"],
            "strengths": evaluation["strengths"],
            "gaps": evaluation["gaps"],
            "interview_questions": evaluation["interview_questions"],
            "role": parsed_resume.get("primary_role_title", ""),
            "domain": parsed_resume.get("primary_domain", ""),
            "error": "",
            "current_step": "evaluation_complete"
        }
        
    except Exception as e:
        print(f"Self evaluation agent error: {str(e)}")
        update = {
            "error": f"Evaluation failed: {str(e)[:100]}",
            "current_step": "evaluation_failed",
            "score_100": 0,
            "strengths": [],
            "gaps": [],
            "interview_questions": []
        }
    
    return {**state, **update}

def self_eval_decision_agent(state: SelfEvalState) -> SelfEvalState:
    """Node 2: The 'brain' that decides what to do next based on score."""
    print("--- AGENT 2: Decision Agent ---")
    
    score = state.get("score_100", 0)
    
    if score < 60:
        # 🔴 Needs confidence + fundamentals
        next_action = "confidence_feedback"
        decision_reasoning = f"Score {score} is below 60. Candidate needs confidence building and fundamental skill improvement. Will generate basic interview questions and a 4-8 week learning roadmap."
        state["interview_questions"] = []
    elif 60 <= score < 80:
        # 🟡 Almost ready, needs targeted improvement
        next_action = "gap_analysis"
        decision_reasoning = f"Score {score} is between 60-79. Candidate is almost ready but has specific gaps. Will generate targeted interview questions and a 2-4 week learning roadmap."
    else:
        # 🟢 Interview ready, needs advanced prep
        next_action = "interview_prep"
        decision_reasoning = f"Score {score} is 80 or above. Candidate is interview-ready. Will generate advanced interview questions and focus on interview preparation."
    
    return {
        **state,
        "next_action": next_action,
        "decision_reasoning": decision_reasoning,
        "current_step": "decision_made"
    }

def confidence_feedback_agent(state: SelfEvalState) -> SelfEvalState:
    """Specialized agent for low-scoring candidates (< 50)."""
    print("--- AGENT 3A: Confidence Feedback Agent ---")
    
    score = state.get("score_100", 0)
    strengths = state.get("strengths", [])
    gaps = state.get("gaps", [])
    
    # Generate confidence-building feedback
    feedback = f"""
Based on your current score of {score}, here's how to build confidence:

Your strengths:
{chr(10).join(f'• {s}' for s in strengths[:3])}

Focus areas for the next 4-8 weeks:
1. Master the fundamentals mentioned in the job description
2. Practice explaining your projects clearly
3. Build 1-2 small projects to demonstrate core skills

Remember: Many successful candidates start where you are now. 
Focus on consistent daily practice rather than perfection.
"""
    
    # Generate basic interview questions
    basic_questions = [
        "Tell me about your most relevant project experience.",
        "What are your key strengths for this role?",
        "How do you approach learning new technologies?",
        "Can you walk me through a technical challenge you solved?",
        "Where do you see yourself improving in the next 3 months?"
    ]
    
    # Generate learning roadmap
    roadmap = [
        "Weeks 1-2: Master fundamentals through online courses",
        "Weeks 3-4: Build a small project using required skills",
        "Weeks 5-6: Practice explaining your project in simple terms",
        "Weeks 7-8: Mock interviews focusing on fundamentals"
    ]
    
    return {
        **state,
        "confidence_feedback": feedback.strip(),
        "learning_roadmap": roadmap,
        "interview_questions": [],  # No questions if score < 60
        "next_focus": "Strengthen fundamentals before interview preparation.",
        "current_step": "confidence_feedback_complete"
    }

def gap_analysis_agent(state: SelfEvalState) -> SelfEvalState:
    """Specialized agent for medium-scoring candidates (50-79)."""
    print("--- AGENT 3B: Gap Analysis Agent ---")
    
    score = state.get("score_100", 0)
    gaps = state.get("gaps", [])
    
    # Generate targeted feedback
    feedback = f"""
With a score of {score}, you're close to being interview-ready!

Your main gaps to address:
{chr(10).join(f'• {gap}' for gap in gaps[:4])}

Targeted action plan:
1. Pick the top 2 gaps and create a focused study plan
2. Practice explaining how you're addressing these gaps
3. Prepare stories that show growth in these areas
"""
    
    # Generate gap-based interview questions
    gap_questions = []
    for gap in gaps[:3]:
        gap_questions.append(f"How would you approach a task requiring {gap}?")
        gap_questions.append(f"What experience do you have with {gap}, and how are you improving?")
    
    # Generate learning roadmap
    roadmap = [
        "Week 1: Deep dive into the top 2 identified gaps",
        "Week 2: Build a mini-project addressing these gaps",
        "Week 3: Practice behavioral questions about skill growth",
        "Week 4: Mock interviews focusing on gap areas"
    ]
    
    return {
        **state,
        "confidence_feedback": feedback.strip(),
        "learning_roadmap": roadmap,
        "interview_questions": state.get("interview_questions", []) + gap_questions[:4],
        "current_step": "gap_analysis_complete"
    }

def interview_questions_agent(state: SelfEvalState) -> SelfEvalState:
    """Specialized agent for high-scoring candidates (≥ 80)."""
    print("--- AGENT 3C: Interview Questions Agent ---")
    
    score = state.get("score_100", 0)
    role = state.get("role", "")
    domain = state.get("domain", "")
    
    # Generate advanced interview preparation
    feedback = f"""
Excellent! With a score of {score}, you're well-prepared for {role} roles in {domain}.

Advanced preparation strategy:
1. Focus on system design and architecture questions
2. Prepare detailed stories about complex projects
3. Practice leadership and mentoring scenarios
4. Research the company's specific tech stack deeply
"""
    
    # Generate advanced interview questions
    advanced_questions = [
        f"Design a scalable system for a key {domain} application.",
        f"How would you mentor a junior developer on your team?",
        f"Describe a time you had to make a tough technical decision.",
        f"What emerging trends in {domain} are you most excited about?",
        f"How do you balance technical debt with feature development?"
    ]
    
    # Generate focused learning roadmap
    roadmap = [
        "Week 1: Deep dive into company's tech stack and projects",
        "Week 2: Practice system design and architecture questions",
        "Week 3: Prepare leadership and behavioral stories",
        "Week 4: Mock interviews with senior engineers"
    ]
    
    return {
        **state,
        "confidence_feedback": feedback.strip(),
        "learning_roadmap": roadmap,
        "interview_questions": state.get("interview_questions", []) + advanced_questions,
        "current_step": "interview_prep_complete"
    }

def error_handling_agent(state: SelfEvalState) -> SelfEvalState:
    """Handles errors gracefully and ensures safe response."""
    print("--- AGENT ERROR: Error Handling ---")
    
    error = state.get("error", "Unknown error")
    
    # Provide safe defaults
    return {
        **state,
        "confidence_feedback": f"System encountered an issue: {error[:100]}. Basic evaluation completed.",
        "learning_roadmap": ["Review the job description requirements", "Practice common interview questions"],
        "decision_reasoning": f"Error occurred: {error[:50]}... Using fallback strategy.",
        "current_step": "error_handled"
    }

def route_agentic_workflow(state: SelfEvalState) -> str:
    """Routes to the appropriate specialized agent based on decision."""
    
    if state.get("error"):
        print("ROUTE: Error detected, routing to error handler")
        return "error_handler"
    
    next_action = state.get("next_action", "")
    
    if next_action == "confidence_feedback":
        return "confidence_agent"
    elif next_action == "gap_analysis":
        return "gap_agent"
    elif next_action == "interview_prep":
        return "interview_agent"
    else:
        print(f"ROUTE: Unknown action '{next_action}', defaulting to gap analysis")
        return "gap_agent"

def route_after_specialized(state: SelfEvalState) -> str:
    """Routes after specialized agent completes."""
    if state.get("error"):
        return "error_handler"
    return END

# =====================================================
#          AGENTIC SELF-EVALUATION: WORKFLOW
# =====================================================

# Create the agentic self-evaluation workflow
self_eval_workflow = StateGraph(SelfEvalState)

# Add nodes
self_eval_workflow.add_node("evaluate", self_eval_agent)
self_eval_workflow.add_node("decide", self_eval_decision_agent)
self_eval_workflow.add_node("confidence_agent", confidence_feedback_agent)
self_eval_workflow.add_node("gap_agent", gap_analysis_agent)
self_eval_workflow.add_node("interview_agent", interview_questions_agent)
self_eval_workflow.add_node("error_handler", error_handling_agent)

# Set entry point
self_eval_workflow.set_entry_point("evaluate")

# Add edges
self_eval_workflow.add_edge("evaluate", "decide")

# Add conditional routing after decision
self_eval_workflow.add_conditional_edges(
    "decide",
    route_agentic_workflow,
    {
        "confidence_agent": "confidence_agent",
        "gap_agent": "gap_agent",
        "interview_agent": "interview_agent",
        "error_handler": "error_handler"
    }
)

# Add edges from specialized agents to END
self_eval_workflow.add_edge("confidence_agent", END)
self_eval_workflow.add_edge("gap_agent", END)
self_eval_workflow.add_edge("interview_agent", END)
self_eval_workflow.add_edge("error_handler", END)

# Compile the workflow
agentic_self_eval_app = self_eval_workflow.compile()

# =====================================================
#          ORIGINAL RESUME PARSING WORKFLOW
# =====================================================

class ResumeParserState(TypedDict):
    """State for the resilient resume parsing workflow."""
    pdf_bytes: bytes
    resume_text: str
    parsed_data: Optional[Dict[str, Any]]
    embedding: Optional[List[float]]
    file_name: str
    error_message: Optional[str]
    max_retries: int
    retry_count: int
    storage_result: Optional[Dict]

def extract_node(state: ResumeParserState) -> ResumeParserState:
    """Node 1: Extracts text from PDF bytes."""
    print("--- 1. Running PDF Extraction ---")
    try:
        text = extract_pdf_text(state["pdf_bytes"])
        return {"resume_text": text, "error_message": None}
    except Exception as e:
        return {"error_message": f"PDF Extraction Failed: {str(e)}"}


def parse_node(state: ResumeParserState) -> ResumeParserState:
    """Node 2: Parses text into structured JSON using the LLM (with retry)."""
    current_retry = state.get("retry_count", 0) + 1
    print(f"--- 2. Running LLM Parsing (Attempt {current_retry}) ---")
    
    prompt_modifier = ""
    if state["error_message"]:
        prompt_modifier = f"ATTENTION: Previous attempt failed due to invalid JSON or missing data ({state['error_message'][:50]}...). MUST return VALID JSON ONLY, ensuring all required fields are present."

    try:
        parsed_data = parse_resume_simple(state["resume_text"], prompt_modifier) 
        
        if not parsed_data.get("candidate_name"):
             raise ValueError("Parsed data is missing a critical 'candidate_name' field.")
             
        return {"parsed_data": parsed_data, "error_message": None, "retry_count": current_retry}
    
    except Exception as e:
        return {"error_message": f"Parsing Failed: {str(e)}", "retry_count": current_retry}

def embed_and_store_node(state: ResumeParserState) -> ResumeParserState:
    """Node 3: Generates embedding and stores data in Supabase."""
    print("--- 3. Running Embedding and Storage ---")
    
    try:
        candidate_text = generate_candidate_text_for_embedding(state["parsed_data"])
        embedding = embed_text(candidate_text)
        
        storage_result = store_resume_in_supabase(state["file_name"], state["parsed_data"], embedding)
        
        return {"embedding": embedding, "error_message": None, "storage_result": storage_result}

    except Exception as e:
        return {"error_message": f"Storage/Embedding Failed: {str(e)}"}

def route_parsing(state: ResumeParserState) -> str:
    """Decides whether to retry parsing, fail, or continue to storage."""
    
    if state.get("parsed_data"):
        print("ROUTE: Parsing successful. Proceeding to Embed/Store.")
        return "store"
    
    if state.get("error_message") and state.get("retry_count", 0) < state.get("max_retries", 3):
        print(f"ROUTE: Parsing failed. Retrying... (Count: {state['retry_count']}/{state['max_retries']})")
        return "parse" 
    else:
        print(f"ROUTE: Parsing/Extraction failed. Ending workflow.")
        return "failed"

workflow = StateGraph(ResumeParserState)

workflow.add_node("extract", extract_node)
workflow.add_node("parse", parse_node)
workflow.add_node("store", embed_and_store_node)

workflow.set_entry_point("extract")
workflow.add_edge("extract", "parse")

workflow.add_conditional_edges(
    "parse",
    route_parsing,
    {
        "store": "store",
        "parse": "parse",
        "failed": END,
    }
)

workflow.add_edge("store", END)

app_workflow = workflow.compile()

# =====================================================
#                       ROUTES
# =====================================================

@app.get("/")
async def serve_frontend():
    """Serve the frontend"""
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        test = supabase.table("candidates_parsed").select("count", count="exact").limit(1).execute()
        count = test.count or 0
        
        return {
            "status": "healthy",
            "supabase_connected": True,
            "candidates_in_db": count,
            "message": f"Found {count} resumes in database",
            "timestamp": datetime.utcnow().isoformat(),
            "system_version": "1.5.0",
            "features": ["LangGraph", "pgvector", "Metrics API", "Groq LLM", "Gemini Embeddings", "Agentic Self-Evaluation", "Interview Evaluation Agent"]
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "supabase_connected": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# =====================================================
#          NEW INTERVIEW EVALUATION ENDPOINT
# =====================================================

@app.post("/interview/evaluate", response_model=InterviewEvalResponse)
async def evaluate_interview_answer(request: InterviewAnswerRequest):
    """
    Evaluate an interview answer using AI agent.
    """
    try:
        # Create initial state
        initial_state = InterviewEvalState(
            question=request.question,
            user_answer=request.user_answer,
            role=request.role or "",
            domain=request.domain or "",
            answer_score=0,
            feedback="",
            strengths=[],
            weaknesses=[],
            follow_up_question="",
            next_difficulty="same",
            error=""
        )
        
        # Run the interview evaluation workflow
        final_state = interview_eval_app.invoke(initial_state)
        
        # Determine status based on error
        status = "failed" if final_state.get("error") else "success"
        
        # Return structured response
        return InterviewEvalResponse(
            answer_score=final_state.get("answer_score", 0),
            feedback=final_state.get("feedback", "Unable to evaluate answer."),
            strengths=final_state.get("strengths", []),
            weaknesses=final_state.get("weaknesses", []),
            follow_up_question=final_state.get("follow_up_question", "Can you explain this in another way?"),
            next_difficulty=final_state.get("next_difficulty", "same"),
            status=status
        )
        
    except Exception as e:
        # Catch-all to prevent 500 errors
        print(f"Interview evaluation endpoint error: {str(e)}")
        return InterviewEvalResponse(
            answer_score=0,
            feedback=f"System error occurred during evaluation: {str(e)[:100]}",
            strengths=[],
            weaknesses=["System error occurred"],
            follow_up_question="Can you explain this in another way?",
            next_difficulty="same",
            status="failed"
        )

# =====================================================
#          NEW AGENTIC SELF-EVALUATION ENDPOINT
# =====================================================

@app.post("/agent-self-evaluation", response_model=AgentSelfEvalResponse)
async def agent_self_evaluation(
    jd_text: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Agentic self-evaluation: Uses LangGraph workflow with intelligent decision-making.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        pdf_bytes = await file.read()
        
        # Initialize state with safe defaults
        initial_state = SelfEvalState(
            pdf_bytes=pdf_bytes,
            jd_text=jd_text,
            resume_text="",
            score_100=0,
            strengths=[],
            gaps=[],
            interview_questions=[],
            role="",
            domain="",
            learning_roadmap=[],
            confidence_feedback="",
            next_action="",
            decision_reasoning="",
            error="",
            current_step="started",
            parsed_resume={},
            candidate_text=""
        )
        
        # Run the agentic workflow
        final_state = agentic_self_eval_app.invoke(initial_state)
        
        # Determine status based on workflow outcome
        if final_state.get("error"):
            status = "partial" if final_state.get("score_100", 0) > 0 else "failed"
        else:
            status = "success"
        
        # Always return a complete response (never 500)
        return AgentSelfEvalResponse(
            score_100=final_state.get("score_100", 0),
            strengths=final_state.get("strengths", []),
            gaps=final_state.get("gaps", []),
            interview_questions=final_state.get("interview_questions", []),
            learning_roadmap=final_state.get("learning_roadmap", []),
            confidence_feedback=final_state.get("confidence_feedback", "Evaluation completed."),
            decision_reasoning=final_state.get("decision_reasoning", "Workflow completed."),
            status=status
        )
        
    except Exception as e:
        # Catch-all to prevent 500 errors
        print(f"Agent self-evaluation endpoint error: {str(e)}")
        return AgentSelfEvalResponse(
            score_100=0,
            strengths=[],
            gaps=[],
            interview_questions=[],
            learning_roadmap=["Review job description", "Practice common questions"],
            confidence_feedback=f"System encountered an issue: {str(e)[:100]}",
            decision_reasoning="Fallback response due to system error",
            status="failed"
        )

# =====================================================
#          ORIGINAL SELF-EVALUATION ENDPOINT (UNCHANGED)
# =====================================================

@app.post("/self-evaluation", response_model=SelfEvalResponse)
async def self_evaluation(
    jd_text: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Self-evaluation: Compare user's own resume against a job description
    (Original non-agentic version - unchanged)
    """
    try:
        pdf_bytes = await file.read()
        resume_text = extract_pdf_text(pdf_bytes)
        
        parsed = parse_resume_simple(resume_text)
        
        candidate_text = generate_candidate_text_for_embedding(parsed)
        
        # Use Groq for evaluation
        evaluation = evaluate_candidate_with_groq(jd_text, candidate_text)
        
        return SelfEvalResponse(
            score_100=evaluation["score_100"],
            strengths=evaluation["strengths"],
            gaps=evaluation["gaps"],
            interview_questions=evaluation["interview_questions"],
            role=parsed.get("primary_role_title"),
            domain=parsed.get("primary_domain")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Self-evaluation failed: {str(e)}")

# =====================================================
#          EXISTING ENDPOINTS (UNCHANGED)
# =====================================================

@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """
    Upload a new resume - utilizes the LangGraph workflow for resilient parsing.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
    pdf_bytes = await file.read()
    
    initial_state = ResumeParserState(
        pdf_bytes=pdf_bytes,
        resume_text="",
        parsed_data=None,
        embedding=None,
        file_name=file.filename,
        error_message=None,
        max_retries=3,
        retry_count=0,
        storage_result=None
    )
    
    try:
        final_state = app_workflow.invoke(initial_state)
        
        if final_state.get("error_message") or not final_state.get("parsed_data"):
            error_msg = final_state.get("error_message", "Unknown parsing failure.")
            raise RuntimeError(f"Workflow failed after max retries or critical error: {error_msg}")

        parsed = final_state["parsed_data"]
        storage_result = final_state["storage_result"]
        embedding_dim = len(final_state["embedding"]) if final_state["embedding"] else 0
        
        return {
            "status": "success",
            "message": "Resume parsed and stored successfully (LangGraph Flow)",
            "candidate_id": storage_result["candidate_id"],
            "parsed_data": {
                "name": parsed.get("candidate_name"),
                "role": parsed.get("primary_role_title"),
                "domain": parsed.get("primary_domain"),
                "experience": parsed.get("total_experience_years"),
                "skills": [s.get("skill_name") for s in parsed.get("skills", [])]
            },
            "parsed_full": parsed,
            "embedding": {
                "dimensions": embedding_dim,
                "stored_in": "pgvector"
            }
        }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """
    Get comprehensive system metrics for the frontend dashboard.
    """
    try:
        # Get all metrics in parallel (would be async in production)
        db_metrics = get_database_metrics()
        score_dist = calculate_score_distribution()
        top_skills = get_top_skills(limit=10)
        monthly_activity = get_monthly_activity_data()
        performance_metrics = get_performance_metrics()
        recent_uploads = get_recent_uploads(limit=10)
        
        # Extract domain distribution for separate use
        domain_dist = db_metrics.get("domain_distribution", {})
        domain_labels = list(domain_dist.keys())[:7]
        domain_data = list(domain_dist.values())[:7]
        
        # Calculate key metrics
        total_candidates = db_metrics.get("total_candidates", 0)
        recent_uploads_7d = db_metrics.get("recent_uploads_7d", 0)
        avg_experience = db_metrics.get("avg_experience_years", 0)
        unique_skills = db_metrics.get("unique_skills", 0)
        
        # Calculate trends (simulated for now)
        weekly_growth = min(15, max(5, int(recent_uploads_7d / max(1, total_candidates) * 100)))
        avg_match_score = 78.5  # Simulated - would come from evaluations table
        avg_processing_time = 2.4  # Simulated - would come from performance logs
        
        return MetricsResponse(
            key_metrics={
                "total_candidates": total_candidates,
                "avg_match_score": avg_match_score,
                "parsed_resumes": total_candidates,  # Same as total_candidates
                "avg_processing_time": avg_processing_time,
                "weekly_growth_percent": weekly_growth,
                "avg_experience_years": avg_experience,
                "unique_skills": unique_skills
            },
            score_distribution={
                "labels": score_dist.get("labels", []),
                "datasets": [{"data": score_dist.get("data", [])}]
            },
            top_skills={
                "labels": top_skills.get("labels", []),
                "datasets": [{"data": top_skills.get("data", [])}]
            },
            domain_distribution={
                "labels": domain_labels,
                "datasets": [{"data": domain_data}]
            },
            monthly_activity=monthly_activity,
            performance_metrics=performance_metrics,
            database_stats={
                "embeddings_stored": total_candidates,
                "uptime_percentage": performance_metrics.get("uptime_percentage", 99.8),
                "storage_used_gb": round(total_candidates * 0.005, 2),  # Simulated: 5MB per candidate
                "api_calls_today": recent_uploads_7d * 3,  # Simulated: 3 API calls per upload
                "active_candidates_last_7d": recent_uploads_7d,
                "total_skills_count": unique_skills
            },
            recent_uploads=recent_uploads,
            top_domains=[
                {"domain": domain, "count": count} 
                for domain, count in sorted(domain_dist.items(), key=lambda x: x[1], reverse=True)[:5]
            ]
        )
        
    except Exception as e:
        print(f"Error in metrics endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch metrics: {str(e)}")

@app.get("/metrics/summary")
async def get_metrics_summary():
    """Get a quick summary of key metrics"""
    try:
        db_metrics = get_database_metrics()
        
        return {
            "total_candidates": db_metrics.get("total_candidates", 0),
            "recent_uploads_7d": db_metrics.get("recent_uploads_7d", 0),
            "unique_skills": db_metrics.get("unique_skills", 0),
            "avg_experience_years": db_metrics.get("avg_experience_years", 0),
            "top_domains": list(db_metrics.get("domain_distribution", {}).keys())[:3]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch metrics summary: {str(e)}")

@app.post("/evaluate-jd", response_model=EvaluateJDResponse)
async def evaluate_jd(request: EvaluateJDRequest):
    """
    Evaluate a job description against your 66+ resumes using pgvector similarity
    """
    try:
        jd_embedding = embed_text(request.jd_text)
        
        similar_candidates = search_candidates_by_similarity(
            query_embedding=jd_embedding,
            top_k=request.top_k,
            domain=request.domain
        )
        
        if not similar_candidates:
            raise HTTPException(status_code=404, detail="No candidates found in database")
        
        results = []
        for candidate in similar_candidates:
            candidate_text = f"""
Name: {candidate.get('candidate_name') or 'N/A'}
Role: {candidate.get('primary_role_title') or 'N/A'}
Domain: {candidate.get('primary_domain') or 'N/A'}
Experience: {candidate.get('total_experience_years') or 'N/A'} years
Summary: {candidate.get('summary_text') or 'N/A'}
Skills: {candidate.get('skills_text') or 'N/A'}
""".strip()
            
            # Use Groq for evaluation
            evaluation = evaluate_candidate_with_groq(request.jd_text, candidate_text)
            
            results.append(CandidateEval(
                candidate_id=candidate["id"],
                candidate_name=candidate.get("candidate_name"),
                primary_role=candidate.get("primary_role_title"),
                primary_domain=candidate.get("primary_domain"),
                total_experience=candidate.get("total_experience_years"),
                score_100=evaluation["score_100"],
                strengths=evaluation["strengths"],
                gaps=evaluation["gaps"],
                interview_questions=evaluation["interview_questions"]
            ))
        
        return EvaluateJDResponse(
            jd_text=request.jd_text,
            domain_filter=request.domain,
            results=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/candidates/count")
async def get_candidate_count():
    """Get count of candidates in database"""
    try:
        result = supabase.table("candidates_parsed")\
            .select("count", count="exact")\
            .execute()
        
        return {
            "total_candidates": result.count or 0,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/candidates/sample")
async def get_sample_candidates(limit: int = 5):
    """Get sample of candidates from database"""
    try:
        result = supabase.table("candidates_parsed")\
            .select("id, candidate_name, primary_role_title, primary_domain, total_experience_years")\
            .limit(limit)\
            .execute()
        
        return {
            "candidates": result.data or [],
            "count": len(result.data) if result.data else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/skills/popular")
async def get_popular_skills(limit: int = 20):
    """Get most popular skills from database"""
    try:
        result = supabase.table("candidate_skills")\
            .select("skill_name")\
            .execute()
        
        if not result.data:
            return {"skills": [], "total": 0}
        
        skills = [skill["skill_name"] for skill in result.data]
        skill_counts = Counter(skills)
        
        popular_skills = [
            {"skill": skill, "count": count}
            for skill, count in skill_counts.most_common(limit)
        ]
        
        return {
            "skills": popular_skills,
            "total_unique": len(skill_counts),
            "total_mentions": len(skills)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/activity/daily")
async def get_daily_activity(days: int = 7):
    """Get daily upload activity"""
    try:
        daily_activity = []
        
        for i in range(days):
            day = datetime.utcnow() - timedelta(days=i)
            day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day.replace(hour=23, minute=59, second=59, microsecond=999999)
            
            result = supabase.table("candidates_parsed")\
                .select("count", count="exact")\
                .gte("created_at", day_start.isoformat())\
                .lt("created_at", day_end.isoformat())\
                .execute()
            
            daily_activity.append({
                "date": day.strftime("%Y-%m-%d"),
                "uploads": result.count or 0
            })
        
        daily_activity.reverse()
        
        return {
            "period_days": days,
            "activity": daily_activity,
            "total_uploads": sum(item["uploads"] for item in daily_activity)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
#                       MAIN
# =====================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("ATS Resume Intelligence Platform v1.5.0")
    print("Now with Interview Evaluation Agent")
    print(f"Supabase URL: {SUPABASE_URL[:30]}...")
    print("="*60)
    print("\nEndpoints:")
    print("  GET  /                     - Frontend")
    print("  GET  /health              - Health check")
    print("  GET  /metrics             - Comprehensive metrics dashboard")
    print("  GET  /metrics/summary     - Quick metrics summary")
    print("  GET  /skills/popular      - Most popular skills")
    print("  GET  /activity/daily      - Daily upload activity")
    print("  POST /upload-resume       - Upload & parse resume (LangGraph)")
    print("  POST /evaluate-jd         - Match JD against 66+ resumes")
    print("  POST /self-evaluation     - Personal fit evaluation (original)")
    print("  POST /agent-self-evaluation - Agentic self-evaluation")
    print("  POST /interview/evaluate  - Interview answer evaluation (NEW)")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )