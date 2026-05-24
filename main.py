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

# Initialize the limiter using the user's remote IP address
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()

# Attach the limiter to FastAPI's error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enable CORS so your desktop index.html can talk to Railway
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
    return {"message": "Toy Identifier API is running with Rate Limiting enabled!"}

# We add our dual limits here: 3 per hour AND 10 per day
@app.post("/identify")
@limiter.limit("3/hour;10/day")
async def identify_toy(request: Request, file: UploadFile = File(...)):
    print(f"📦 Received scan request from IP: {get_remote_address(request)}")
    
    # 1. Read the raw image bytes
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    except Exception as e:
        print(f"❌ Error reading file bytes: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to read file.")

    # 2. Call the Gemini Vision API
    try:
        print("🤖 Sending image bytes over to Gemini...")
        client = genai.Client()
        
        response = client.models.generate_content(
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
        raise HTTPException(status_code=400, detail=f"Gemini processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
