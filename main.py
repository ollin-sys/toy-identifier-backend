import os
import json
import uvicorn
import contextvars
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from google import genai
from google.genai import types

# --- Pydantic Model ---
class FigureIdentity(BaseModel):
    character_name: str
    toy_line: str
    manufacturer: str
    release_year: str
    variant_details: str
    confidence_score: float

# --- Middleware & Limiter Setup ---
_current_request = contextvars.ContextVar("current_request")

class RequestStateMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        token = _current_request.set(request)
        try:
            return await call_next(request)
        finally:
            _current_request.reset(token)

def is_premium_user() -> bool:
    try:
        request: Request = _current_request.get()
    except LookupError:
        return False
    user_tier = request.headers.get("x-user-tier", "free").lower()
    handshake_token = request.headers.get("x-auth-token", "")
    master_key = os.environ.get("FIGSEEKER_PREMIUM_KEY", "FIGSEEKER_PREMIUM_KEY_Pa$$1")
    return user_tier == "premium" and handshake_token == master_key

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_middleware(RequestStateMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- API Routes ---
@app.post("/identify")
@limiter.limit("3/hour;10/day", exempt_when=is_premium_user)
async def identify_toy(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        client = genai.Client()
        response = await client.aio.models.generate_content(
            model='gemini-2.0-flash',
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=file.content_type or "image/jpeg"),
                "Identify this action figure or toy. Provide details matching the structural schema."
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FigureIdentity,
            ),
        )
        return JSONResponse(content=json.loads(response.text))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini processing failed: {str(e)}")

# --- Static Files (Must be last) ---
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
