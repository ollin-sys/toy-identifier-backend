import os
import json
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

# Import the rate limiting tools
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Helper to capture the current active request thread context
from starlette.middleware.base import BaseHTTPMiddleware
import contextvars

# Context variable to hold the request globally for the slowapi callback rule
_current_request = contextvars.ContextVar("current_request")

class RequestStateMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        token = _current_request.set(request)
        try:
            response = await call_next(request)
            return response
        finally:
            _current_request.reset(token)

# 1. Zero-argument callback function satisfying slowapi's internal requirements
def is_premium_user() -> bool:
    """
    Validates if the active context request genuinely qualifies for Premium tier bypass.
    """
    try:
        request: Request = _current_request.get()
    except LookupError:
        print("⚠️ slowapi exemption check could not resolve the ContextVar Request object.")
        return False

    user_tier = request.headers.get("x-user-tier", "free").lower()
    handshake_token = request.headers.get("x-auth-token", "")
    
    # Matches the exact variable name and has a solid fallback for safety
    master_premium_key = os.environ.get("FIGSEEKER_PREMIUM_KEY", "FIGSEEKER_PREMIUM_KEY_Pa$$1")
    
    if user_tier == "premium" and handshake_token == master_premium_key:
        print("✨ Premium Handshake Verified! Exempting from rate limits.")
        return True
        
    return False

# Initialize the limiter cleanly using the user's remote IP address
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()

# Register the ContextVar middleware first
app.add_middleware(RequestStateMiddleware)

# Attach the limiter to FastAPI's error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enable CORS so your domain can communicate safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FigureIdentity(BaseModel):
    character_name: str
    toy_line: str
    manufacturer: str
    release_year: str
    variant_details: str
    confidence_score: float

# --- MODIFIED HOME ROUTE TO SERVE FRONTEND DASHBOARD ---
@app.get("/")
def home():
    """
    Reads and delivers the index.html user interface page 
    automatically whenever the primary backend root URL is loaded.
    """
    return FileResponse("index.html")

# Public rate limits: 3 per hour AND 10 per day handled safely by slowapi decorator
@app.post("/identify")
@limiter.limit("3/hour;10/day", exempt_when=is_premium_user)
async def identify_toy(request: Request, file: UploadFile = File(...)):
    print(f"📦 Scan request received from IP: {get_remote_address(request)}")
    
    # 2. READ THE RAW IMAGE BYTES
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    except Exception as e:
        print(f"❌ Error reading file bytes: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to read file.")

    # 3. CALL THE GEMINI VISION API ASYNCHRONOUSLY
    try:
        if not os.environ.get("GEMINI_API_KEY"):
            print("❌ CRITICAL: GEMINI_API_KEY variable is missing from the environment!")
            raise HTTPException(status_code=500, detail="Server configuration error: Missing API Key.")

        print("🤖 Initializing Google GenAI Client...")
        client = genai.Client()
        
        print("🤖 Sending image bytes over to Gemini async channel via client.aio...")
        response = await client.aio.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=file.content_type or "image/jpeg",
                ),
                "Identify this action figure, LEGO minifigure, or toy. Provide details matching the structural schema."
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FigureIdentity,
            ),
        )
        print("✅ Gemini successfully returned structured data string!")
        
        structured_json = json.loads(response.text)
        return JSONResponse(content=structured_json)
        
    except Exception as e:
        print(f"❌ Gemini API Error crash: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Gemini processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
