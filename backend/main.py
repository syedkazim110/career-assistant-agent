from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
import logging
from typing import Optional
from dotenv import load_dotenv

from models.schemas import (
    AnalysisResult,
    JobAnalysis,
    ResumeAnalysis,
    SkillGap,
    GeneratedDocument,
    UploadResponse,
    EmailRequest,
    EmailResponse,
    AgentRequest,
    AgentResponse,
    EvaluationRequest,
    EvaluationResponse
)
from services.pdf_parser import PDFParser
from services.gemini_service import GeminiService
from services.skill_analyzer import SkillAnalyzer
from services.document_generator import DocumentGenerator
from services.email_service import EmailService
from services.agent_service import CareerAgentService
from services.agent_evaluator import AgentEvaluator, BenchmarkComparison

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get base directory for resolving paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize FastAPI app
app = FastAPI(
    title="Automated Career Assistant API",
    description="AI-powered resume tailoring and cover letter generation",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables")

gemini_service = GeminiService(GEMINI_API_KEY) if GEMINI_API_KEY else None
pdf_parser = PDFParser()
skill_analyzer = SkillAnalyzer()
document_generator = DocumentGenerator(output_dir=os.path.join(BASE_DIR, "generated"))

# In-memory storage for analysis results (in production, use a database)
analysis_store = {}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Automated Career Assistant API",
        "status": "running",
        "version": "1.0.0"
    }

@app.post("/api/upload-and-analyze", response_model=AnalysisResult)
async def upload_and_analyze(
    resume: UploadFile = File(...),
    job_description: UploadFile = File(...)
):
    """
    Upload resume and job description PDFs, then analyze them
    
    Args:
        resume: Resume PDF file
        job_description: Job description PDF file
        
    Returns:
        Analysis results with skill gap and match percentage
    """
    try:
        # Validate file types
        if not resume.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Resume must be a PDF file")
        if not job_description.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Job description must be a PDF file")
        
        # Generate unique IDs for files
        resume_id = f"resume_{uuid.uuid4().hex}.pdf"
        job_id = f"job_{uuid.uuid4().hex}.pdf"
        
        # Save uploaded files
        resume_path = os.path.join("uploads", resume_id)
        job_path = os.path.join("uploads", job_id)
        
        with open(resume_path, "wb") as f:
            f.write(await resume.read())
        
        with open(job_path, "wb") as f:
            f.write(await job_description.read())
        
        # Extract text from PDFs
        logger.info("Extracting text from PDFs...")
        resume_text = await pdf_parser.extract_text_from_pdf(resume_path)
        job_text = await pdf_parser.extract_text_from_pdf(job_path)
        
        if not resume_text:
            raise HTTPException(status_code=400, detail="Could not extract text from resume")
        if not job_text:
            raise HTTPException(status_code=400, detail="Could not extract text from job description")
        
        # Check if Gemini service is available
        if not gemini_service:
            raise HTTPException(status_code=500, detail="AI service not configured. Please set GEMINI_API_KEY")
        
        # Analyze documents with Gemini
        logger.info("Analyzing documents with AI...")
        job_analysis_data = await gemini_service.analyze_job_description(job_text)
        resume_analysis_data = await gemini_service.analyze_resume(resume_text)
        
        # Create analysis objects
        job_analysis = JobAnalysis(**job_analysis_data)
        resume_analysis = ResumeAnalysis(**resume_analysis_data)
        
        # Analyze skill gap
        logger.info("Analyzing skill gaps...")
        all_job_skills = job_analysis.required_skills + job_analysis.preferred_skills
        skill_gap_data = skill_analyzer.analyze_skill_gap(
            all_job_skills,
            resume_analysis.skills
        )
        
        skill_gap = SkillGap(**skill_gap_data)
        
        # Calculate match percentage
        match_percentage = skill_analyzer.calculate_match_percentage(
            len(skill_gap.matching_skills),
            len(skill_gap.partial_skills),
            len(all_job_skills)
        )
        
        # Create analysis result
        analysis_result = AnalysisResult(
            job_analysis=job_analysis,
            resume_analysis=resume_analysis,
            skill_gap=skill_gap,
            match_percentage=match_percentage
        )
        
        # Store analysis and file paths for later use
        analysis_id = uuid.uuid4().hex
        analysis_store[analysis_id] = {
            "result": analysis_result,
            "resume_path": resume_path,
            "job_path": job_path,
            "resume_text": resume_text,
            "job_text": job_text
        }
        
        # Clean up uploaded files (keep them for document generation)
        # os.remove(resume_path)
        # os.remove(job_path)
        
        logger.info(f"Analysis completed. Match: {match_percentage}%")
        return analysis_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in upload_and_analyze: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@app.post("/api/generate-resume")
async def generate_resume(
    resume: UploadFile = File(...),
    job_description: UploadFile = File(...),
    format: str = Form("docx")
):
    """
    Generate a tailored resume based on job requirements
    
    Args:
        resume: Resume PDF file
        job_description: Job description PDF file
        format: Output format (docx or pdf)
        
    Returns:
        Generated resume file
    """
    try:
        # Validate format
        if format not in ["docx", "pdf"]:
            raise HTTPException(status_code=400, detail="Format must be 'docx' or 'pdf'")
        
        # Generate unique IDs for files
        resume_id = f"resume_{uuid.uuid4().hex}.pdf"
        job_id = f"job_{uuid.uuid4().hex}.pdf"
        
        # Save uploaded files
        resume_path = os.path.join("uploads", resume_id)
        job_path = os.path.join("uploads", job_id)
        
        with open(resume_path, "wb") as f:
            f.write(await resume.read())
        
        with open(job_path, "wb") as f:
            f.write(await job_description.read())
        
        # Extract text from PDFs
        logger.info("Extracting text from PDFs...")
        resume_text = await pdf_parser.extract_text_from_pdf(resume_path)
        job_text = await pdf_parser.extract_text_from_pdf(job_path)
        
        if not resume_text or not job_text:
            raise HTTPException(status_code=400, detail="Could not extract text from files")
        
        # Check if Gemini service is available
        if not gemini_service:
            raise HTTPException(status_code=500, detail="AI service not configured")
        
        # Generate tailored resume content
        logger.info("Generating tailored resume...")
        tailored_content = await gemini_service.generate_tailored_resume(
            resume_text,
            job_text
        )
        
        # Use fixed filename and delete old ones
        filename = f"latest_resume.{format}"
        file_path_to_delete = os.path.join(BASE_DIR, "generated", filename)
        
        # Delete old resume if exists
        if os.path.exists(file_path_to_delete):
            os.remove(file_path_to_delete)
            logger.info(f"Deleted old resume: {filename}")
        
        # Generate new document
        if format == "docx":
            file_path = document_generator.generate_docx_resume(tailored_content, filename)
        else:
            file_path = document_generator.generate_pdf_resume(tailored_content, filename)
        
        # Clean up uploaded files
        os.remove(resume_path)
        os.remove(job_path)
        
        logger.info(f"Resume generated: {filename}")
        
        # Return file
        return FileResponse(
            file_path,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document" if format == "docx" else "application/pdf",
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating resume: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating resume: {str(e)}")

@app.post("/api/generate-cover-letter")
async def generate_cover_letter(
    resume: UploadFile = File(...),
    job_description: UploadFile = File(...),
    format: str = Form("docx")
):
    """
    Generate a cover letter based on resume and job description
    
    Args:
        resume: Resume PDF file
        job_description: Job description PDF file
        format: Output format (docx or pdf)
        
    Returns:
        Generated cover letter file
    """
    try:
        # Validate format
        if format not in ["docx", "pdf"]:
            raise HTTPException(status_code=400, detail="Format must be 'docx' or 'pdf'")
        
        # Generate unique IDs for files
        resume_id = f"resume_{uuid.uuid4().hex}.pdf"
        job_id = f"job_{uuid.uuid4().hex}.pdf"
        
        # Save uploaded files
        resume_path = os.path.join("uploads", resume_id)
        job_path = os.path.join("uploads", job_id)
        
        with open(resume_path, "wb") as f:
            f.write(await resume.read())
        
        with open(job_path, "wb") as f:
            f.write(await job_description.read())
        
        # Extract text from PDFs
        logger.info("Extracting text from PDFs...")
        resume_text = await pdf_parser.extract_text_from_pdf(resume_path)
        job_text = await pdf_parser.extract_text_from_pdf(job_path)
        
        if not resume_text or not job_text:
            raise HTTPException(status_code=400, detail="Could not extract text from files")
        
        # Check if Gemini service is available
        if not gemini_service:
            raise HTTPException(status_code=500, detail="AI service not configured")
        
        # Generate cover letter content
        logger.info("Generating cover letter...")
        cover_letter_content = await gemini_service.generate_cover_letter(
            resume_text,
            job_text
        )
        
        # Use fixed filename and delete old ones
        filename = f"latest_cover_letter.{format}"
        file_path_to_delete = os.path.join(BASE_DIR, "generated", filename)
        
        # Delete old cover letter if exists
        if os.path.exists(file_path_to_delete):
            os.remove(file_path_to_delete)
            logger.info(f"Deleted old cover letter: {filename}")
        
        # Generate new document
        if format == "docx":
            file_path = document_generator.generate_docx_cover_letter(cover_letter_content, filename)
        else:
            file_path = document_generator.generate_pdf_cover_letter(cover_letter_content, filename)
        
        # Clean up uploaded files
        os.remove(resume_path)
        os.remove(job_path)
        
        logger.info(f"Cover letter generated: {filename}")
        
        # Return file
        return FileResponse(
            file_path,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document" if format == "docx" else "application/pdf",
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating cover letter: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating cover letter: {str(e)}")

@app.post("/api/send-email", response_model=EmailResponse)
async def send_application_email(request: EmailRequest):
    """
    Send job application email with resume and cover letter attachments
    
    Args:
        request: Email request with recipient, subject, body, and file paths
        
    Returns:
        EmailResponse with success status
    """
    try:
        # Get email configuration from environment
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        sender_email = os.getenv("SENDER_EMAIL")
        sender_password = os.getenv("SENDER_PASSWORD")
        
        if not sender_email or not sender_password:
            raise HTTPException(
                status_code=500,
                detail="Email not configured. Please set SENDER_EMAIL and SENDER_PASSWORD"
            )
        
        # Initialize email service
        email_service = EmailService(
            smtp_server=smtp_server,
            smtp_port=smtp_port,
            sender_email=sender_email,
            sender_password=sender_password
        )
        
        # Validate recipient email
        if not email_service.validate_email(request.recipient_email):
            raise HTTPException(status_code=400, detail="Invalid recipient email address")
        
        # Prepare attachment paths - resolve relative to BASE_DIR
        attachments = []
        
        # Helper function to resolve paths
        def resolve_path(path: str) -> Optional[str]:
            """Resolve file path relative to BASE_DIR"""
            if not path:
                return None
            
            # Try the path as-is first
            if os.path.isabs(path) and os.path.exists(path):
                return path
            
            # Try relative to BASE_DIR
            base_path = os.path.join(BASE_DIR, path)
            if os.path.exists(base_path):
                return base_path
            
            # Try without 'backend/' prefix if present
            if path.startswith('backend/'):
                alt_path = os.path.join(BASE_DIR, path[8:])
                if os.path.exists(alt_path):
                    return alt_path
            
            logger.warning(f"File not found at path: {path} (tried: {path}, {base_path})")
            return None
        
        # Resolve resume path
        if request.resume_path:
            resume_abs_path = resolve_path(request.resume_path)
            if resume_abs_path:
                attachments.append(resume_abs_path)
                logger.info(f"Resume attachment found: {resume_abs_path}")
        
        # Resolve cover letter path
        if request.cover_letter_path:
            cover_abs_path = resolve_path(request.cover_letter_path)
            if cover_abs_path:
                attachments.append(cover_abs_path)
                logger.info(f"Cover letter attachment found: {cover_abs_path}")
        
        if not attachments:
            raise HTTPException(
                status_code=400, 
                detail=f"No valid attachments found. Resume path: {request.resume_path}, Cover letter path: {request.cover_letter_path}"
            )
        
        # Send email
        logger.info(f"Sending application email to {request.recipient_email}")
        success = await email_service.send_application_email(
            recipient_email=request.recipient_email,
            subject=request.subject,
            body=request.body,
            attachment_paths=attachments
        )
        
        if success:
            return EmailResponse(
                success=True,
                message="Application email sent successfully!"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to send email")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error sending email: {str(e)}")

# Initialize agent service (lazy loading)
career_agent = None

def get_agent():
    """Get or create agent service"""
    global career_agent
    if career_agent is None and GEMINI_API_KEY:
        career_agent = CareerAgentService(
            api_key=GEMINI_API_KEY,
            document_generator=document_generator,
            output_dir=os.path.join(BASE_DIR, "generated")
        )
    return career_agent

@app.post("/api/agent/run", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    """
    Run the Career Assistant Agent with a natural language task
    
    This endpoint demonstrates Track A compliance by:
    - Using LangChain tools for real actions
    - Autonomous decision-making about which tools to use
    - Executing actual operations (file parsing, document generation, etc.)
    
    Args:
        request: Agent request with task description
        
    Returns:
        Agent response with output and action history
    """
    try:
        agent = get_agent()
        if not agent:
            raise HTTPException(
                status_code=500,
                detail="Agent not configured. Please set GEMINI_API_KEY"
            )
        
        logger.info(f"Running agent with task: {request.task}")
        
        # Run agent
        result = await agent.run(request.task)
        
        return AgentResponse(
            success=result.get("success", False),
            output=result.get("output", ""),
            intermediate_steps=result.get("intermediate_steps"),
            action_history=result.get("action_history"),
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent execution error: {str(e)}")

@app.get("/api/agent/metrics")
async def get_agent_metrics():
    """
    Get agent performance metrics
    
    Returns agent performance statistics for evaluation purposes
    """
    try:
        agent = get_agent()
        if not agent:
            return {"message": "Agent not initialized yet"}
        
        metrics = agent.get_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting agent metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving metrics: {str(e)}")

@app.post("/api/agent/evaluate", response_model=EvaluationResponse)
async def evaluate_agent(request: EvaluationRequest):
    """
    Run agent evaluation suite
    
    This endpoint satisfies Track A evaluation requirements by:
    - Running comprehensive test cases
    - Measuring task success rate
    - Evaluating tool usage efficiency
    - Generating detailed performance reports
    
    Args:
        request: Evaluation request parameters
        
    Returns:
        Evaluation response with results summary
    """
    try:
        agent = get_agent()
        if not agent:
            raise HTTPException(
                status_code=500,
                detail="Agent not configured. Please set GEMINI_API_KEY"
            )
        
        logger.info("Starting agent evaluation...")
        
        # Create evaluator
        evaluator = AgentEvaluator(agent)
        
        # Run evaluation
        report = await evaluator.evaluate_all(output_dir=request.output_dir)
        
        # Generate benchmark comparison
        benchmark = BenchmarkComparison.generate_benchmark_report(report)
        
        # Prepare summary
        summary = {
            "timestamp": report.timestamp,
            "total_tests": report.total_tests,
            "passed_tests": report.passed_tests,
            "failed_tests": report.failed_tests,
            "success_rate": report.success_rate,
            "average_steps": report.average_steps,
            "average_execution_time": report.average_execution_time,
            "benchmark": benchmark
        }
        
        # Save path
        report_filename = f"evaluation_report_{report.timestamp.replace(':', '-').split('.')[0]}.json"
        report_path = os.path.join(request.output_dir, report_filename) if request.save_results else None
        
        return EvaluationResponse(
            success=True,
            report_path=report_path,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        return EvaluationResponse(
            success=False,
            error=str(e)
        )

@app.get("/api/agent/tools")
async def list_agent_tools():
    """
    List available agent tools
    
    Returns information about all tools the agent can use
    """
    try:
        agent = get_agent()
        if not agent:
            return {"message": "Agent not initialized yet", "tools": []}
        
        tools_info = [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in agent.tools
        ]
        
        return {
            "total_tools": len(tools_info),
            "tools": tools_info,
            "framework": "LangChain with Google Gemini",
            "agent_type": "ReAct (Reasoning + Acting)"
        }
        
    except Exception as e:
        logger.error(f"Error listing tools: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing tools: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "gemini_configured": gemini_service is not None,
        "agent_available": GEMINI_API_KEY is not None,
        "track_a_compliant": True,
        "features": {
            "langchain_integration": True,
            "tool_system": True,
            "agent_evaluation": True,
            "real_actions": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
