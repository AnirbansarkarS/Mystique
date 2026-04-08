from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os

from prompt_builder import build_system_prompt, load_profile

app = FastAPI()

PROFILE_FILE = "personality_profile.json"

class FeedbackRequest(BaseModel):
    message: str
    feedback: str # "sounds like her" | "not quite"

class ChatRequest(BaseModel):
    user_message: str

def save_profile(profile: dict, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=4)

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

@app.post("/api/chat")
def chat(request: ChatRequest):
    """
    Simulates sending a message to the AI with the formulated prompt.
    """
    if not os.path.exists(PROFILE_FILE):
        raise HTTPException(status_code=404, detail="Profile not found")

    profile = load_profile(PROFILE_FILE)
    system_prompt = build_system_prompt(profile)

    # In a real app, send `system_prompt` and `request.user_message` to Claude/OpenAI API here.
    
    return {
        "system_prompt": system_prompt,
        "simulated_llm_response": "Baba, ki korcho akhon? Rat 1 ta baje, tui jekane giyechish fire aye!"
    }
