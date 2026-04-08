from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import json
import os
import logging
import google.generativeai as genai

from prompt_builder import build_system_prompt, load_profile

logging.basicConfig(level=logging.INFO)

app = FastAPI()

PROFILE_FILE = "personality_profile.json"

class FeedbackRequest(BaseModel):
    message: str
    feedback: str # "sounds like her" | "not quite"

class ChatRequest(BaseModel):
    user_message: str
    api_key: str

def save_profile(profile: dict, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=4)

@app.get("/")
def serve_frontend():
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/api/feedback")
def submit_feedback(request: FeedbackRequest):
    """
    Saves good examples to profile based on feedback.
    """
    if not os.path.exists(PROFILE_FILE):
        raise HTTPException(status_code=404, detail="Profile not found")

    profile = load_profile(PROFILE_FILE)

    if request.feedback == "sounds like her":
        if "good_examples" not in profile:
            profile["good_examples"] = []
        if request.message not in profile["good_examples"]:
            profile["good_examples"].append(request.message)
            save_profile(profile, PROFILE_FILE)
            return {"status": "success", "message": "Example saved to profile"}
    
    return {"status": "ignored", "message": "Feedback not positive, example ignored."}

@app.post("/api/upload_chat")
async def extract_persona_from_chat(file: UploadFile = File(...), target_name: str = Form(...), api_key: str = Form(...)):
    """
    Parses WhatsApp txt, extracts messages by target, uses Gemini to build personality JSON.
    """
    contents = await file.read()
    chat_text = contents.decode("utf-8", errors="ignore")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash") # Use gemini-1.5-flash

    extraction_prompt = f\"\"\"
    Analyze the following chat log to build a personality profile of `{target_name}`. 
    Notice their tone, frequently used vocabulary, and emotional triggers (how they react to specific subjects).
    Return ONLY valid JSON in this exact structure:
    {{
        "vocabulary": ["catchphrase1", "slang2"],
        "tone": "short_and_sweet",
        "avg_reply_len": "short/medium/long",
        "triggers": {{ "subject1": "reaction", "subject2": "reaction" }},
        "good_examples": ["an exact quote from them", "another quote"]
    }}
    
    Chat Log Snippet:
    {chat_text[:30000]}
    \"\"\"

    try:
        res = model.generate_content(extraction_prompt)
        json_resp = res.text.strip()
        if json_resp.startswith("```json"):
            json_resp = json_resp[7:]
        if json_resp.endswith("```"):
            json_resp = json_resp[:-3]
        
        parsed_profile = json.loads(json_resp)
        save_profile(parsed_profile, PROFILE_FILE)
        return {"status": "success", "message": "Profile extracted via Gemini."}
    except Exception as e:
        logging.error(f"Error extracting profile: {{e}}")
        raise HTTPException(status_code=500, detail="Gemini failed to extract JSON profile.")

@app.post("/api/chat")
def chat(request: ChatRequest):
    """
    Simulates sending a message to the AI with the formulated prompt.
    """
    if not os.path.exists(PROFILE_FILE):
        raise HTTPException(status_code=404, detail="Profile not found. Please upload a chat file and train first.")

    profile = load_profile(PROFILE_FILE)
    system_prompt = build_system_prompt(profile)

    try:
        genai.configure(api_key=request.api_key)
        model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=system_prompt)
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(request.user_message)
        
        return {
            "system_prompt": system_prompt,
            "llm_response": response.text.replace("\n", " ").strip()
        }
    except Exception as e:
        logging.error(f"Error querying Gemini: {e}")
        raise HTTPException(status_code=500, detail="Gemini failed to respond. Check API Key.")
