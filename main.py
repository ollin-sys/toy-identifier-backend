import os
import json
import uvicorn
import contextvars
import stripe
from fastapi import FastAPI, UploadFile, File, Request, HTTPException, status
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

# Initialize Stripe
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")

class CheckoutRequest(BaseModel):
    product_id: str

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
    # REMOVED: hardcoded fallback key for security
    master_key = os.environ.get("FIGSEEKER_PREMIUM_KEY")
    
    return user_tier == "premium" and handshake_token == master_key and master_key is not None

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_middleware(RequestStateMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- API Routes ---

@app.post("/create-checkout-session")
async def create_checkout(request: CheckoutRequest):
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': request.product_id,
                'quantity': 1,
            }],
            mode='payment',
            success_url="https://yourdomain.com/success",
            cancel_url="https://yourdomain.com/cancel",
        )
        return {"url": session.url}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Checkout failed")

@app.post("/identify")
@limiter.limit("3/hour;10/day", exempt_when=is_premium_user)
async def identify_toy(request: Request, file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported file format.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    try:
        client = genai.Client()
        response = await client.aio.models.generate_content(
            model='gemini-3.5-flash',
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=file.content_type),
                "Identify this action figure or toy. Provide details matching the structural schema."
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FigureIdentity,
            ),
        )
        return JSONResponse(content=json.loads(response.text))
    except Exception as e:
        # Log 'e' here in a real production environment
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Identification service unavailable.")

# --- Static Files ---
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
