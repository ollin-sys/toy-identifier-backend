import os
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

# Enable CORS so your desktop index.html can talk to Railway
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define our structured output schema using Pydantic
class FigureIdentity(BaseModel):
    character_name: str
    toy_line: str
    manufacturer: str
    release_year: str
    variant_details: str
    confidence_score: float

@app.get("/")
def home():
    return {"message": "Toy Identifier API is running!"}

@app.post("/identify")
async def identify_toy(file: UploadFile = File(...)):
    print(f"📦 Received file upload: {file.filename}, content_type: {file.content_type}")
    
    # 1. Read the raw image bytes
    try:
        image_bytes = await file.read()
        if not image_bytes:
            print("❌ Error: The uploaded file is empty.")
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    except Exception as e:
        print(f"❌ Error reading file bytes: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to read file.")

    # 2. Call the Gemini Vision API and parse structured output
    try:
        print("🤖 Sending image bytes over to Gemini...")
        client = genai.Client() # Picks up your GEMINI_API_KEY env variable automatically
        
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
        
        # Parse the structural response text string into native JSON object
        structured_json = json.loads(response.text)
        
        # Return the clean object wrapped in a FastAPI JSONResponse
        return JSONResponse(content=structured_json)
        
    except Exception as e:
        print(f"❌ Gemini API Error crash: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Gemini processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
