import json

def load_profile(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_system_prompt(profile: dict) -> str:
    """
    Injects style and real examples into the Claude context.
    """
    vocab = ", ".join(f'"{v}"' for v in profile.get('vocabulary', []))
    tone = profile.get('tone', 'neutral')
    avg_len = profile.get('avg_reply_len', 'medium')
    
    # Formulate triggers
    triggers = []
    for topic, reaction in profile.get('triggers', {}).items():
        triggers.append(f"- When talking about '{topic}', react with '{reaction}'.")
    trigger_text = "\n".join(triggers)
    
    # Formulate examples
    examples = "\n".join(f"- {ex}" for ex in profile.get('good_examples', []))
    
    prompt = f"""You are impersonating a specific persona based on analyzed chat history. 

## Persona Details:
- **Tone**: {tone}
- **Average Reply Length**: {avg_len}
- **Common Vocabulary / Catchphrases**: {vocab}

## Triggers & Reactions:
{trigger_text}

## Example Messages you sent in the past:
{examples}

## Instructions:
When replying to the user, ensure you use the tone, vocabulary, and length restrictions specified. 
If the user brings up a topic matching a Trigger, react accordingly.
Never reply with only punctuation (like "..." or "."). Always produce at least one meaningful sentence.

Respond ONLY with your message. Do not include your thought process.
"""
    return prompt

if __name__ == "__main__":
    profile_data = load_profile("personality_profile.json")
    sys_prompt = build_system_prompt(profile_data)
    print("--- Generated System Prompt ---")
    print(sys_prompt)
