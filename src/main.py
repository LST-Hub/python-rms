from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import csv
import pandas as pd
import io
import PyPDF2
from io import BytesIO, StringIO
import tempfile
import os
import time
from datetime import datetime
from dotenv import load_dotenv

from src.llm.resume_extractor import ResumeExtractor
from src.llm.candidate_fit import CandidateFitEvaluator
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

load_dotenv()

# Pydantic models
class UserIn(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class DownloadRequest(BaseModel):
    data: Dict[str, Any]
    format: str

class JobDescriptionRequest(BaseModel):
    job_description: str

class CandidateFitRequest(BaseModel):
    resume_data: List[Dict[str, Any]]
    job_description_data: str
    fit_options: Optional[Dict[str, Any]] = None

app = FastAPI(title="Resume Extractor API", version="3.0.0-Stateless")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple rate limiting without persistent state
class SimpleRateLimiter:
    def __init__(self, max_requests_per_minute=100):
        self.max_requests = max_requests_per_minute
        self.requests = []
    
    def check_rate_limit(self):
        """Simple rate limiting - clears old requests and checks current count"""
        now = time.time()
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]
        
        if len(self.requests) >= self.max_requests:
            return False
        
        self.requests.append(now)
        return True

# Initialize components
resume_extractor = ResumeExtractor()
candidate_fit_evaluator = CandidateFitEvaluator()
rate_limiter = SimpleRateLimiter(max_requests_per_minute=100)

def process_single_resume_sync(resume_extractor, file_path, filename):
    """Process a single resume synchronously"""
    try:
        # Verify file exists before processing
        if not os.path.exists(file_path):
            return {"filename": filename, "error": "File not found", "success": False}
        
        # Extract resume data
        result = resume_extractor.extract_from_file(file_path, filename)
        
        if result is None:
            print(f"Warning: No result for {filename}")
            return {"filename": filename, "error": "No data extracted", "success": False}
        
        return result
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return {"filename": filename, "error": str(e), "success": False}
    finally:
        # Clean up temp file
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error cleaning up temp file {file_path}: {str(e)}")

def process_single_candidate_fit_sync(evaluator, resume, job_description, fit_options=None):
    """Process a single candidate fit evaluation synchronously"""
    try:
        result = evaluator.evaluate_fit(resume, job_description, fit_options)
        
        if result:
            if 'candidate_name' not in result:
                result['candidate_name'] = resume.get('personal_info', {}).get('name', 'Unknown')

            # Ensure all required fields are present
            if 'fit_percentage' not in result:
                result['fit_percentage'] = 0
            if 'summary' not in result:
                result['summary'] = "Analysis failed to produce summary."
            if 'key_matches' not in result:
                result['key_matches'] = []
            if 'key_gaps' not in result:
                result['key_gaps'] = []
                
            return result
        else:
            candidate_name = resume.get('personal_info', {}).get('name', 'Unknown')
            return {
                "candidate_name": candidate_name,
                "fit_percentage": 0,
                "summary": "Could not evaluate candidate fit.",
                "key_matches": [],
                "key_gaps": [],
                "error": "Evaluation failed to return results",
                "success": False
            }
        
    except Exception as e:
        candidate_name = resume.get('personal_info', {}).get('name', 'Unknown')
        print(f"Error evaluating {candidate_name}: {str(e)}")
        return {
            "candidate_name": candidate_name,
            "fit_percentage": 0,
            "summary": "Error during evaluation.",
            "key_matches": [],
            "key_gaps": [],
            "error": str(e),
            "success": False
        }

@app.get("/")
async def root():
    return {"message": "Resume Extractor API v3.0 is running in stateless mode"}

@app.post("/extract-resume")
async def extract_resume_data(files: List[UploadFile] = File(...)):
    """
    Extract data from uploaded resume files synchronously
    """
    try:
        # Rate limiting check
        if not rate_limiter.check_rate_limit():
            raise HTTPException(
                status_code=429, 
                detail="Rate limit exceeded. Please try again later."
            )
        
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        # Limit number of files per request for memory management
        if len(files) > 50:  # Reduced limit for stateless processing
            raise HTTPException(status_code=400, detail="Too many files. Maximum 50 files per request.")
        
        results = []
        processed_count = 0
        
        print(f"Processing {len(files)} files synchronously")
        
        for file in files:
            try:
                # Check file size
                if file.size > 10 * 1024 * 1024:  # 10MB limit
                    print(f"File {file.filename} too large: {file.size} bytes")
                    results.append({
                        "filename": file.filename,
                        "error": "File too large (>10MB)",
                        "success": False
                    })
                    continue
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
                    content = await file.read()
                    tmp_file.write(content)
                    tmp_file_path = tmp_file.name
                
                # Process the resume
                result = process_single_resume_sync(resume_extractor, tmp_file_path, file.filename)
                results.append(result)
                processed_count += 1
                
                print(f"Processed {processed_count}/{len(files)}: {file.filename}")
                
            except Exception as e:
                print(f"Error processing file {file.filename}: {str(e)}")
                results.append({
                    "filename": file.filename,
                    "error": str(e),
                    "success": False
                })
                continue
        
        successful_extractions = sum(1 for r in results if r.get('success', True) and 'error' not in r)
        print(f"Completed processing: {successful_extractions}/{len(files)} successful extractions")
        
        return {
            'success': True,
            'extracted_data': results,
            'total_files': len(files),
            'successful_extractions': successful_extractions,
            'processing_time': 'immediate'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in extract_resume_data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/extract-job-description")
async def extract_job_description(request: JobDescriptionRequest):
    """Process job description text"""
    try:
        if not request.job_description or not request.job_description.strip():
            raise HTTPException(status_code=400, detail="Job description is required")
        
        job_description_data = request.job_description.strip()
        
        return {
            "success": True,
            "job_description_data": job_description_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing job description: {str(e)}")

@app.post("/candidate-fit")
async def candidate_fit(request: CandidateFitRequest):
    """Compare multiple resumes and job description synchronously"""
    try:
        # Rate limiting check
        if not rate_limiter.check_rate_limit():
            raise HTTPException(
                status_code=429, 
                detail="Rate limit exceeded. Please try again later."
            )
        
        if not request.resume_data:
            raise HTTPException(status_code=400, detail="No resume data provided")
        
        if not request.job_description_data:
            raise HTTPException(status_code=400, detail="Job description is required")
        
        # Limit for memory management
        if len(request.resume_data) > 30:  # Reduced limit for stateless processing
            raise HTTPException(status_code=400, detail="Too many resumes. Maximum 30 resumes per request.")
        
        fit_options = request.fit_options or {}
        results = []
        processed_count = 0
        
        print(f"Processing candidate fit for {len(request.resume_data)} resumes")
        
        for resume in request.resume_data:
            try:
                # Make sure resume is a dictionary, not a list
                if isinstance(resume, list):
                    resume = resume[0] if resume else {}
                
                result = process_single_candidate_fit_sync(
                    candidate_fit_evaluator, 
                    resume, 
                    request.job_description_data, 
                    fit_options
                )
                results.append(result)
                processed_count += 1
                
                candidate_name = result.get('candidate_name', 'Unknown')
                print(f"Processed {processed_count}/{len(request.resume_data)}: {candidate_name}")
                
            except Exception as e:
                print(f"Error processing resume: {str(e)}")
                results.append({
                    "candidate_name": "Unknown",
                    "fit_percentage": 0,
                    "summary": "Error during evaluation.",
                    "key_matches": [],
                    "key_gaps": [],
                    "error": str(e),
                    "success": False
                })
                continue
        
        # Sort results by fit percentage (descending)
        results.sort(key=lambda x: x.get('fit_percentage', 0), reverse=True)
        
        successful_evaluations = sum(1 for r in results if r.get('success', True) and 'error' not in r)
        print(f"Completed candidate fit analysis: {successful_evaluations}/{len(request.resume_data)} successful evaluations")
        
        return {
            "success": True,
            "fit_results": results,
            "total_resumes": len(request.resume_data),
            "successful_evaluations": successful_evaluations,
            "processing_time": "immediate"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in candidate_fit: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/download-data")
async def download_data(request: DownloadRequest):
    """Convert extracted data to requested format and return as downloadable file"""
    try:
        data = request.data
        format_type = request.format.lower()
        
        if format_type == "json":
            return create_json_response(data)
        elif format_type == "csv":
            return create_csv_response(data)
        elif format_type in ["excel", "xlsx"]:
            return create_excel_response(data)
        elif format_type == "pdf":
            return create_pdf_response(data)
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating download: {str(e)}")

@app.post("/download-fit-excel")
async def download_fit_excel(data: dict):
    """Download candidate fit results as Excel file"""
    try:
        fit_results = data.get("fit_results", [])
        if not fit_results:
            raise HTTPException(status_code=400, detail="No fit results provided.")

        # Flatten and prepare data for DataFrame
        rows = []
        for idx, fit in enumerate(fit_results, 1):
            rows.append({
                "Rank": idx,
                "Candidate Name": fit.get("candidate_name", f"Candidate {idx}"),
                "Fit Percentage": fit.get("fit_percentage", ""),
                "Summary": fit.get("summary", ""),
                "Key Matches": ", ".join(fit.get("key_matches", [])),
                "Key Gaps": ", ".join(fit.get("key_gaps", [])),
            })
        
        df = pd.DataFrame(rows)
        output = io.BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=candidate_fit_results.xlsx"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating Excel file: {str(e)}")

@app.get("/system/health")
async def health_check():
    """System health check for stateless version"""
    try:
        # Test resume extractor
        extractor_status = "healthy" if resume_extractor else "not_initialized"
        
        return {
            "status": "healthy",
            "version": "3.0.0-stateless",
            "resume_extractor": extractor_status,
            "processing_mode": "synchronous",
            "memory_model": "stateless",
            "rate_limiter": {
                "type": "simple",
                "current_requests": len(rate_limiter.requests),
                "max_requests_per_minute": rate_limiter.max_requests
            },
            "limits": {
                "max_files_per_request": 50,
                "max_resumes_for_fit": 30,
                "max_file_size_mb": 10
            },
            "features": {
                "job_storage": False,
                "background_processing": False,
                "progress_tracking": False,
                "async_queues": False
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# File download helper functions (same as before but included for completeness)
def create_json_response(data):
    """Create JSON file response"""
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    
    return StreamingResponse(
        BytesIO(json_str.encode('utf-8')),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=resume_data.json"}
    )

def create_csv_response(data):
    """Create CSV file response"""
    output = StringIO()
    
    # Flatten the data for CSV format
    flattened_data = []
    
    if "extracted_data" in data:
        for resume in data["extracted_data"]:
            flat_resume = {}
            flat_resume["filename"] = resume.get("filename", "")
            
            # Extract basic info
            if "personal_info" in resume:
                for key, value in resume["personal_info"].items():
                    flat_resume[f"personal_{key}"] = value
            
            # Extract skills
            if "skills" in resume:
                if isinstance(resume["skills"], list):
                    flat_resume["skills"] = ", ".join(resume["skills"])
                else:
                    flat_resume["skills"] = str(resume["skills"])
            
            # Extract experience
            if "experience" in resume:
                for i, exp in enumerate(resume["experience"]):
                    if isinstance(exp, dict):
                        for key, value in exp.items():
                            flat_resume[f"experience_{i+1}_{key}"] = value
                    else:
                        flat_resume[f"experience_{i+1}"] = str(exp)
            
            # Extract education
            if "education" in resume:
                for i, edu in enumerate(resume["education"]):
                    if isinstance(edu, dict):
                        for key, value in edu.items():
                            flat_resume[f"education_{i+1}_{key}"] = value
                    else:
                        flat_resume[f"education_{i+1}"] = str(edu)
            
            flattened_data.append(flat_resume)
    
    if flattened_data:
        writer = csv.DictWriter(output, fieldnames=flattened_data[0].keys())
        writer.writeheader()
        writer.writerows(flattened_data)
    
    return StreamingResponse(
        BytesIO(output.getvalue().encode('utf-8')),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=resume_data.csv"}
    )

def create_excel_response(data):
    """Create Excel file response"""
    output = BytesIO()
    
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Create summary sheet
            summary_data = []
            
            if "extracted_data" in data:
                for resume in data["extracted_data"]:
                    summary_row = {
                        "Name": resume.get("personal_info", {}).get("name", ""),
                        "Email": resume.get("personal_info", {}).get("email", ""),
                        "Phone": resume.get("personal_info", {}).get("phone", ""),
                        "Location": resume.get("personal_info", {}).get("location", ""),
                        "Skills": ", ".join(resume.get("skills", [])) if isinstance(resume.get("skills"), list) else str(resume.get("skills", "")),
                        "Experience": "; ".join([
                            f"{exp.get('title', '')} at {exp.get('company', '')} ({exp.get('location', '')}) - {exp.get('duration', '')}"
                            for exp in resume.get("experience", [])
                        ]),
                        "Education": "; ".join([
                            f"{edu.get('degree', '')}"
                            for edu in resume.get("education", [])
                        ]),
                        "Designation": resume.get("experience", [{}])[0].get("title", "") if isinstance(resume.get("experience"), list) and resume.get("experience") else "",
                        "Summary": resume.get("summary", ""),
                        "Total Experience": resume.get("total_experience", ""),
                        "Certifications": "; ".join([
                            f"{cert.get('name', '')}"
                            for cert in resume.get("certifications", [])
                        ]),
                        "Languages": "; ".join([
                            f"{lang.get('language', '')}"
                            for lang in resume.get("languages", [])
                        ]),
                        "Awards": resume.get("awards", ""),
                        "Projects": "; ".join([
                            f"{proj.get('name', '')}: {proj.get('description', '')}"
                            for proj in resume.get("projects", [])
                        ])
                    }
                    summary_data.append(summary_row)
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Create detailed sheet
            detailed_data = []
            if "extracted_data" in data:
                for resume in data["extracted_data"]:
                    detailed_row = {
                        "filename": resume.get("filename", ""),
                        "full_data": json.dumps(resume, indent=2, ensure_ascii=False)
                    }
                    detailed_data.append(detailed_row)
            
            if detailed_data:
                detailed_df = pd.DataFrame(detailed_data)
                detailed_df.to_excel(writer, sheet_name='Detailed', index=False)
        
        output.seek(0)
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=resume_data.xlsx"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating Excel file: {str(e)}")

def create_pdf_response(data):
    """Create PDF report response"""
    output = BytesIO()
    doc = SimpleDocTemplate(output, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("Resume Extraction Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Summary
    if "extracted_data" in data:
        summary_text = f"Total Resumes Processed: {len(data['extracted_data'])}"
        summary = Paragraph(summary_text, styles['Heading2'])
        story.append(summary)
        story.append(Spacer(1, 12))
        
        # Individual resume details
        for i, resume in enumerate(data["extracted_data"], 1):
            # Resume header
            resume_title = Paragraph(f"Resume {i}: {resume.get('filename', 'Unknown')}", styles['Heading2'])
            story.append(resume_title)
            story.append(Spacer(1, 6))
            
            # Personal info
            if "personal_info" in resume:
                personal_info = resume["personal_info"]
                info_text = f"""
                Name: {personal_info.get('name', 'N/A')}<br/>
                Email: {personal_info.get('email', 'N/A')}<br/>
                Phone: {personal_info.get('phone', 'N/A')}<br/>
                Location: {personal_info.get('location', 'N/A')}
                """
                info_para = Paragraph(info_text, styles['Normal'])
                story.append(info_para)
                story.append(Spacer(1, 6))
            
            # Skills
            if "skills" in resume and resume["skills"]:
                skills_title = Paragraph("Skills:", styles['Heading3'])
                story.append(skills_title)
                if isinstance(resume["skills"], list):
                    skills_text = ", ".join(resume["skills"])
                else:
                    skills_text = str(resume["skills"])
                skills_para = Paragraph(skills_text, styles['Normal'])
                story.append(skills_para)
                story.append(Spacer(1, 6))
            
            # Experience
            if "experience" in resume and resume["experience"]:
                exp_title = Paragraph("Experience:", styles['Heading3'])
                story.append(exp_title)
                for exp in resume["experience"]:
                    if isinstance(exp, dict):
                        exp_text = f"{exp.get('title', 'N/A')} at {exp.get('company', 'N/A')} ({exp.get('duration', 'N/A')})"
                    else:
                        exp_text = str(exp)
                    exp_para = Paragraph(exp_text, styles['Normal'])
                    story.append(exp_para)
                story.append(Spacer(1, 6))
            
            # Education
            if "education" in resume and resume["education"]:
                edu_title = Paragraph("Education:", styles['Heading3'])
                story.append(edu_title)
                for edu in resume["education"]:
                    if isinstance(edu, dict):
                        edu_text = f"{edu.get('degree', 'N/A')} from {edu.get('institution', 'N/A')} ({edu.get('year', 'N/A')})"
                    else:
                        edu_text = str(edu)
                    edu_para = Paragraph(edu_text, styles['Normal'])
                    story.append(edu_para)
                story.append(Spacer(1, 12))
    
    doc.build(story)
    output.seek(0)
    
    return StreamingResponse(
        output,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=resume_report.pdf"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)