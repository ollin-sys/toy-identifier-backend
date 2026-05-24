import os
import json
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

# Import the rate limiting tools
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# 1. Custom Secure Handshake Rule
def is_premium_user(request: Request) -> bool:
    """
    Validates if the incoming request genuinely qualifies for Premium tier bypass.
    Checks for the structural handshake token assigned in Railway environment variables.
    """
    user_tier = request.headers.get("x-user-tier", "free").lower()
    handshake_token = request.headers.get("x-auth-token", "")
    
    # Matches the exact variable name and has a solid fallback for safety
    master_premium_key = os.environ.get("FIGSEEKER_PREMIUM_KEY", "FIGSEEKER_PREMIUM_KEY_Pa$$1")
    
    if user_tier == "premium" and handshake_token == master_premium_key:
        return True
    return False

# Initialize the limiter cleanly using the user's remote IP address
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()

# Attach the limiter to FastAPI's error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enable CORS so your desktop dashboard or domain can communicate safely
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

@app.get("/")
def home():
    return {"message": "Toy Identifier API running securely in production!"}

# Public rate limits: 3 per hour AND 10 per day
@app.post("/identify")
@limiter.limit("3/hour;10/day", exempt_when=is_premium_user)
async def identify_toy(request: Request, file: UploadFile = File(...)):
    print(f"📦 Scan request received from IP: {get_remote_address(request)}")
    
    # Read the raw image bytes
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    except Exception as e:
        print(f"❌ Error reading file bytes: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to read file.")

    # Call the Gemini Vision API asynchronously
    try:
        print("🤖 Initializing Asynchronous Gemini Client...")
        
        # Check for API key explicitly to prevent low-level driver crashes
        if not os.environ.get("GEMINI_API_KEY"):
            print("❌ CRITICAL: GEMINI_API_KEY variable is missing from the environment!")
            raise HTTPException(status_code=500, detail="Server configuration error: Missing API Key.")

        # Switch to ClientAsync to preserve FastAPI's event loop
        client = genai.ClientAsync()
        
        print("🤖 Sending image bytes over to Gemini async channel...")
        response = await client.models.generate_content(
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
