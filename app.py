


from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import logging
import re
import random
from collections import Counter
import google.generativeai as genai

from prompt_builder import build_system_prompt, load_profile

# logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    Fast algorithm to parse WhatsApp txt locally without hitting Gemini.
    Extracts messages by target and builds the personality JSON.
    """
    try:
        contents = await file.read()
        chat_text = contents.decode("utf-8", errors="ignore")

        # Fast algorithmic extraction line by line
        # Match WhatsApp typical format like: "12/03/23, 14:30 - TargetName: The message"
        # Or simple "TargetName: The message"
        pattern = re.compile(rf"{re.escape(target_name)}:\s*(.*)", re.IGNORECASE)
        lines = chat_text.split('\n')
        
        messages = []
        for line in lines:
            match = pattern.search(line)
            if match:
                msg = match.group(1).strip()
                if msg and "<Media omitted>" not in msg and "http" not in msg:
                    messages.append(msg)

        # Fallback if the name format wasn't exactly matched by the primary regex
        if not messages:
            # Grabs the text after the *last* colon in a line as a desperate guesswork
            potential_msgs = [line.split(":")[-1].strip() for line in lines if ":" in line]
            messages = [m for m in potential_msgs if m and "<Media omitted>" not in m][:250]
            
            # Super fallback if entirely empty file
            if not messages:
                messages = ["Hello.", "Okay.", "I see."]

        # Calculate average length
        avg_words = sum(len(m.split()) for m in messages) / len(messages)
        if avg_words < 5:
            avg_len = "short"
        elif avg_words < 12:
            avg_len = "medium"
        else:
            avg_len = "long"

        # Tally vocabulary (words with 4+ chars)
        all_words = []
        for msg in messages:
            words = re.findall(r'\b\w{4,}\b', msg.lower())
            all_words.extend(words)
        
        # Pick top 8 most used words as their "catchphrases"
        vocab = [word for word, count in Counter(all_words).most_common(8)]

        # Sample some good examples
        good_examples = random.sample(messages, min(8, len(messages)))

        # Define some basic generic triggers out-of-the-box (since we dropped LLM parsing here)
        triggers = {
            "late": "question timing",
            "food": "ask if they ate",
            "work": "ask how it's going",
            "tired": "express sympathy"
        }

        parsed_profile = {
            "vocabulary": vocab,
            "tone": "conversational",
            "avg_reply_len": avg_len,
            "triggers": triggers,
            "good_examples": good_examples
        }

        save_profile(parsed_profile, PROFILE_FILE)
        logging.info("Successfully extracted profile locally using fast regex algorithm.")
        return {"status": "success", "message": "Profile extracted via local fast algorithm."}

    except Exception as e:
        logging.error(f"Error extracting profile algorithmically: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to locally analyze chat: {str(e)}")

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
