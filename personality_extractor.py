import json
import re
from typing import List, Dict

def extract_vocabulary(texts: List[str]) -> List[str]:
    # Placeholder for NLP extraction logic (e.g., using TF-IDF or simple frequency counting)
    # Returning some dummy data matching the image
    return ["ki korcho", "kheyecho?"]

def analyze_tone(texts: List[str]) -> str:
    # Placeholder for sentiment/tone analysis
    return "warm_but_stern"

def calculate_avg_reply_length(texts: List[str]) -> str:
    # Placeholder for calculating average word count per message
    return "short"

def extract_triggers(texts: List[str]) -> Dict[str, str]:
    # Placeholder for finding common topics and reactions
    return {
        "late_night": "scold",
        "food": "care"
    }

def process_logs(file_paths: List[str]) -> dict:
    all_texts = []
    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                all_texts.extend(f.readlines())
        except FileNotFoundError:
            print(f"Warning: File {path} not found.")

    profile = {
        "vocabulary": extract_vocabulary(all_texts),
        "tone": analyze_tone(all_texts),
        "avg_reply_len": calculate_avg_reply_length(all_texts),
        "triggers": extract_triggers(all_texts),
        "good_examples": []
    }
    return profile

def save_profile(profile: dict, output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=4)

if __name__ == "__main__":
    # Example usage:
    inputs = [
        "whatsapp_export.txt",
        "sms_backup.txt", 
        "whisper_transcripts.txt"
    ]
    
    extracted_profile = process_logs(inputs)
    save_profile(extracted_profile, "personality_profile.json")
    print("Extracted personality profile and saved to personality_profile.json")
