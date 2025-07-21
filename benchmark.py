import openai
import json
import os
import time
from difflib import SequenceMatcher

# ========== CONFIG ==========
openai.api_key = "sk-..."  # Replace with your key
MODELS = ["gpt-4o", "gpt-3.5-turbo"]
PROMPT_FOLDER = "./prompts"
RESUME_FILE = "./resumes.json"
GROUND_TRUTH_FILE = "./ground_truths.json"
TEMPERATURE = 0
WAIT = 0.7  # Wait between calls to avoid rate limits
# ============================

def load_prompts():
    prompts = {}
    for fname in os.listdir(PROMPT_FOLDER):
        if fname.endswith(".txt"):
            key = fname.replace(".txt", "")
            with open(os.path.join(PROMPT_FOLDER, fname), "r", encoding="utf-8") as f:
                prompts[key] = f.read()
    return prompts

def load_data():
    with open(RESUME_FILE, "r", encoding="utf-8") as f:
        resumes = json.load(f)
    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        ground_truths = json.load(f)
    return resumes, ground_truths

def extract(model, prompt_template, resume_text):
    try:
        prompt = prompt_template.replace("{{resume_text}}", resume_text)
        response = openai.ChatCompletion.create(
            model=model,
            temperature=TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        output = response.choices[0].message.content.strip()
        return json.loads(output)
    except Exception as e:
        print(f"[{model}] Error parsing response: {e}")
        return {}

def similarity(a, b):
    return SequenceMatcher(None, str(a), str(b)).ratio()

def evaluate(prediction, ground_truth):
    scores = {}
    for key in ground_truth:
        if key not in prediction:
            scores[key] = 0.0
        else:
            scores[key] = similarity(prediction[key], ground_truth[key])
    return scores

def main():
    prompts = load_prompts()
    resumes, truths = load_data()

    for model in MODELS:
        print(f"\n===== Evaluating {model} =====")
        total_scores = {}
        for i, resume in enumerate(resumes):
            print(f"\n--- Resume {i+1} ---")
            combined_result = {}
            for section, prompt in prompts.items():
                output = extract(model, prompt, resume["text"])
                combined_result[section] = output.get(section, output)  # Support both wrapping and flat keys
                time.sleep(WAIT)

            # Save/compare result
            truth = truths[i]
            scores = evaluate(combined_result, truth)
            for k, v in scores.items():
                total_scores[k] = total_scores.get(k, 0) + v
            print("Scores:", scores)

        print(f"\n🔍 Model: {model}")
        for section, total in total_scores.items():
            avg_score = total / len(resumes)
            print(f"{section}: {avg_score:.2f}")

if __name__ == "__main__":
    main()
