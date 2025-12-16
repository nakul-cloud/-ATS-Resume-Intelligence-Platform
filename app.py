import os
import json
from io import BytesIO
from typing import List, Optional, Any, Dict
from datetime import datetime
from pydantic import BaseModel
from typing_extensions import TypedDict

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

# =====================================================
#                       CONFIG
# =====================================================

GEMINI_API_KEY = " "  # Your key
SUPABASE_URL = " "
SUPABASE_KEY = " "

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment variables")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase credentials not set in environment variables")

genai.configure(api_key=GEMINI_API_KEY)

# Use the same models as your notebook
resume_model = genai.GenerativeModel("gemini-2.5-flash-lite")
evaluate_model = genai.GenerativeModel("gemini-2.5-flash-lite")

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

# =====================================================
#                   HELPER FUNCTIONS
#   (These are used by the LangGraph nodes)
# =====================================================

def extract_pdf_text(pdf_bytes: bytes, max_chars: int = 6000) -> str:
    """Extract text from PDF - same as your notebook"""
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

    resp = resume_model.generate_content([prompt, text])

    if not resp.text:
        raise RuntimeError("Gemini returned empty response")

    raw = resp.text.replace("```json", "").replace("```", "").strip()

    print("🔍 GEMINI RAW OUTPUT:\n", raw)

    try:
        return json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Gemini JSON parse failed: {raw}") from e


def generate_candidate_text_for_embedding(parsed: dict) -> str:
    """Generate text for embedding - same logic as your notebook"""
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
    """Generate embedding - same as your notebook"""
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="RETRIEVAL_DOCUMENT",
    )
    return response["embedding"]

# =====================================================
#                   DATABASE OPERATIONS
# =====================================================

def store_resume_in_supabase(file_name: str, parsed: dict, embedding: List[float]) -> Dict:
    # Existing function: Stores parsed data and embedding in Supabase
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
            "embedding": embedding,    # pgvector
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
        # Re-raise as a generic Exception for LangGraph state update
        raise Exception(f"Failed to store in Supabase: {str(e)}")


def search_candidates_by_similarity(
    query_embedding: List[float],
    top_k: int = 5,
    domain: Optional[str] = None
) -> List[Dict]:
    """
    Search candidates using pgvector similarity.
    Domain is a SOFT signal (ranking boost), not a hard filter.
    """
    try:
        payload = {
            "query_embedding": query_embedding,
            "match_count": top_k,
            "domain_filter": domain  # can be None
        }

        result = supabase.rpc("match_candidates", payload).execute()

        if result.data:
            return result.data

        # If RPC returns empty, fallback
        print("pgvector search returned no results, using fallback")
        return search_candidates_fallback(top_k=top_k)

    except Exception as e:
        print("pgvector search failed:", e)
        return search_candidates_fallback(top_k=top_k)



def search_candidates_fallback(top_k: int = 5) -> List[Dict]:
    """
    Safe fallback when pgvector search fails.
    - NO hard domain filtering
    - Returns recent + diverse candidates
    - Keeps downstream ranking code stable
    """
    try:
        result = (
            supabase
            .table("candidates_parsed")
            .select(
                "id, candidate_name, primary_role_title, primary_domain, "
                "total_experience_years, summary_text, skills_text"
            )
            .order("created_at", desc=True)   # 🔑 newest resumes first
            .limit(top_k)
            .execute()
        )

        candidates = result.data or []

        # Add deterministic fallback similarity
        for c in candidates:
            c["similarity"] = 0.5   # neutral similarity score

        return candidates

    except Exception as e:
        print("Fallback search error:", e)
        return []


# =====================================================
#                   LANGGRAPH DEFINITION
# =====================================================

# --- 1. Define the Graph State ---
class ResumeParserState(TypedDict):
    """
    State for the resilient resume parsing workflow.
    """
    pdf_bytes: bytes
    resume_text: str
    parsed_data: Optional[Dict[str, Any]]
    embedding: Optional[List[float]]
    file_name: str
    error_message: Optional[str]
    max_retries: int
    retry_count: int
    storage_result: Optional[Dict]

# --- 2. Define the Nodes ---

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
        # Uses the existing helper function, passing the feedback modifier
        parsed_data = parse_resume_simple(state["resume_text"], prompt_modifier) 
        
        # Critical validation check (moved from FastAPI route to the graph)
        if not parsed_data.get("candidate_name"):
             raise ValueError("Parsed data is missing a critical 'candidate_name' field.")
             
        # Success
        return {"parsed_data": parsed_data, "error_message": None, "retry_count": current_retry}
    
    except Exception as e:
        # Failure: Update error message and increment retry count
        return {"error_message": f"Parsing Failed: {str(e)}", "retry_count": current_retry}


def embed_and_store_node(state: ResumeParserState) -> ResumeParserState:
    """Node 3: Generates embedding and stores data in Supabase."""
    print("--- 3. Running Embedding and Storage ---")
    
    try:
        # Generate embedding
        candidate_text = generate_candidate_text_for_embedding(state["parsed_data"])
        embedding = embed_text(candidate_text)
        
        # Store in Supabase
        storage_result = store_resume_in_supabase(state["file_name"], state["parsed_data"], embedding)
        
        # Success
        return {"embedding": embedding, "error_message": None, "storage_result": storage_result}

    except Exception as e:
        # Failure
        return {"error_message": f"Storage/Embedding Failed: {str(e)}"}

# --- 3. Define the Router (Conditional Edge) ---

def route_parsing(state: ResumeParserState) -> str:
    """Decides whether to retry parsing, fail, or continue to storage."""
    
    if state.get("parsed_data"):
        # Success: Parsed and validated
        print("ROUTE: Parsing successful. Proceeding to Embed/Store.")
        return "store"
    
    if state.get("error_message") and state.get("retry_count", 0) < state.get("max_retries", 3):
        # Failure, but retries remain. Loop back to parse_node.
        print(f"ROUTE: Parsing failed. Retrying... (Count: {state['retry_count']}/{state['max_retries']})")
        return "parse" 
    else:
        # Final failure or critical failure at extraction
        print(f"ROUTE: Parsing/Extraction failed. Ending workflow.")
        return "failed"

# --- 4. Build and Compile the Graph ---

workflow = StateGraph(ResumeParserState)

# Add Nodes
workflow.add_node("extract", extract_node)
workflow.add_node("parse", parse_node)
workflow.add_node("store", embed_and_store_node)

# Set up Edges
workflow.set_entry_point("extract")

# Extraction to Parsing
workflow.add_edge("extract", "parse")

# Parsing to Router
workflow.add_conditional_edges(
    "parse",
    route_parsing,
    {
        "store": "store", # If successful, go to store node
        "parse": "parse", # If failed and retries remain, loop back to parse
        "failed": END,    # If max retries reached, end
    }
)

# Store to End
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
        # Check Supabase connection
        test = supabase.table("candidates_parsed").select("count", count="exact").limit(1).execute()
        count = test.count or 0
        
        return {
            "status": "healthy",
            "supabase_connected": True,
            "candidates_in_db": count,
            "message": f"Found {count} resumes in database",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "supabase_connected": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """
    Upload a new resume - utilizes the LangGraph workflow for resilient parsing.
    """
    # 1. FastAPI Validation
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
    pdf_bytes = await file.read()
    
    # 2. Initialize the LangGraph State
    initial_state = ResumeParserState(
        pdf_bytes=pdf_bytes,
        resume_text="",
        parsed_data=None,
        embedding=None,
        file_name=file.filename,
        error_message=None,
        max_retries=3, # Configurable maximum attempts for parsing
        retry_count=0,
        storage_result=None
    )
    
    try:
        # 3. Execute the LangGraph workflow
        final_state = app_workflow.invoke(initial_state)
        
        # 4. Check the final state for failure
        if final_state.get("error_message") or not final_state.get("parsed_data"):
            error_msg = final_state.get("error_message", "Unknown parsing failure.")
            raise RuntimeError(f"Workflow failed after max retries or critical error: {error_msg}")

        # 5. Success Path
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
        # Handles exceptions from the graph execution
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# =====================================================
#                       OTHER ROUTES (PRESERVED)
# =====================================================

@app.post("/evaluate-jd", response_model=EvaluateJDResponse)
async def evaluate_jd(request: EvaluateJDRequest):
    """
    Evaluate a job description against your 66+ resumes using pgvector similarity
    (This route remains procedural as it does not require complex decision loops)
    """
    try:
        # Generate embedding for the job description
        jd_embedding = embed_text(request.jd_text)
        
        # Search for similar candidates using pgvector
        similar_candidates = search_candidates_by_similarity(
            query_embedding=jd_embedding,
            top_k=request.top_k,
            domain=request.domain
        )
        
        if not similar_candidates:
            raise HTTPException(status_code=404, detail="No candidates found in database")
        
        # Evaluate each candidate (loop preserved)
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
            response = evaluate_model.generate_content([
                prompt,
                f"JOB DESCRIPTION:\n{request.jd_text}\n\nCANDIDATE:\n{candidate_text}"
            ])
            
            # Parse response
            raw = response.text.replace("```json", "").replace("```", "").strip()
            evaluation = json.loads(raw)
            
            # Add to results
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

@app.post("/self-evaluation", response_model=SelfEvalResponse)
async def self_evaluation(
    jd_text: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Self-evaluation: Compare user's own resume against a job description
    (This route remains procedural as it's a one-off evaluation)
    """
    try:
        # Process uploaded resume
        pdf_bytes = await file.read()
        resume_text = extract_pdf_text(pdf_bytes)
        
        # Parse resume
        parsed = parse_resume_simple(resume_text)
        
        # Prepare candidate text
        candidate_text = generate_candidate_text_for_embedding(parsed)
        
        # Evaluate with Gemini
        prompt = """
You are an expert career coach. Evaluate how well this candidate fits the job description.
Return ONLY valid JSON:
{
  "score_100": 0-100,
  "strengths": [],
  "gaps": [],
  "interview_questions": []
}
"""
        response = evaluate_model.generate_content([
            prompt,
            f"JOB DESCRIPTION:\n{jd_text}\n\nCANDIDATE RESUME:\n{candidate_text}"
        ])
        
        # Parse response
        raw = response.text.replace("```json", "").replace("```", "").strip()
        evaluation = json.loads(raw)
        
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

@app.get("/candidates/count")
async def get_candidate_count():
    """Get count of candidates in database (your 66+ resumes)"""
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

# =====================================================
#                       MAIN
# =====================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("ATS Resume Intelligence Platform (LangGraph Enabled)")
    print(f"Supabase URL: {SUPABASE_URL[:30]}...")
    print("="*60)
    print("\nEndpoints:")
    print("  GET  /                 - Frontend")
    print("  GET  /health           - Health check")
    print("  POST /upload-resume   - Upload & parse resume (NOW USES LANGGRAPH)")
    print("  POST /evaluate-jd    - Match JD against 66+ resumes")
    print("  POST /self-evaluation - Personal fit evaluation")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )