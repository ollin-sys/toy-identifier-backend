import os
from fastapi import FastAPI, File, UploadFile, HTTPException
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
    # 1. Read the uploaded image file
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # 2. Configure the AI prompt and instructions
    system_instruction = (
        "You are an expert action figure archivist. Analyze images of action figures "
        "and provide hyper-specific identification details, paying close attention to paint, "
        "sculpts, and variants."
    )
    
    prompt = "Identify this action figure. Provide the character, toy line, manufacturer, year, and specific variant details."

    # 3. Call the Gemini API
    try:
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=[img, prompt],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=FigureIdentity,
                temperature=0.1
            ),
        )
        # Return the structured JSON directly to the frontend
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
