from openai import OpenAI
import concurrent.futures
from tqdm import tqdm
import math
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client_eval = OpenAI(api_key=OPENAI_API_KEY)

EVAL_SYSTEM_PROMPT = (
        "You are a strict medical evaluator. "
        "You will be given two inputs: "
        "1) The correct answer key"
        "2) A model-generated answer text with reasoning and a chosen answer. "
        "If the model’s answer correctly identifies the correct option, output 1. "
        "If the answer is incorrect, missing, or wrong, output 0. "
        "Only output a single number: 0 or 1. No explanation."
)

def gpt_score(model_ans, actual_ans, attempt_id=0, thresh=3):
    """Recursively retries GPT-4.1 scoring in case of transient errors."""
    if attempt_id >= thresh:
        return 0.0

    user_prompt = f"Correct Answer Option: {actual_ans}\nModel Answer: {model_ans}\n"

    try:
        response = client_eval.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=8,
        )
        raw_output = response.choices[0].message.content.strip()
        return float(raw_output)
    except Exception:
        print("trying again..")
        return gpt_score(model_ans, actual_ans, attempt_id + 1)

def get_model_answer_after_think(text):
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return ""

# 🧩 Helper: build the correct "answer string" (e.g., "B: some description")
def get_correct_answer_string(item):
    correct_label = item["label"][0]
    correct_option = next(
        (opt["content"] for opt in item["options"] if opt["letter"] == correct_label),
        None
    )
    #return f"{correct_label}: {correct_option}" if correct_option else correct_label
    return f"{correct_label}"

def nCk(n, k):
    try:
        n = int(n)
        k = int(k)
        if k > n or k < 0:
            return 0
        return math.comb(n, k)
    except:
        return 0

def pass_at_k_formula(n, c, k):
    if c == 0: 
        return 0.0
    if k > n:
        return 1.0
    return 1 - (nCk(n - c, k) / nCk(n, k))

def score_all_responses(responses, correct_ans, max_workers=20):
    scores = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(gpt_score,
                            get_model_answer_after_think(resp),
                            correct_ans)
            for resp in responses
        ]
        for f in futures:
            try:
                scores.append(f.result())
            except:
                scores.append(0)
    return scores 

def evaluate_pass_at_k(data, ks, max_workers=20):
    aggregated_scores = {k: [] for k in ks}

    for item in tqdm(data, desc="Evaluating"):
        correct_ans = get_correct_answer_string(item)
        responses = item["response"] 

        # list of 0 or 1
        correctness = score_all_responses(responses, correct_ans, max_workers=max_workers)

        c = sum(correctness)      
        n = len(correctness)      

        for k in ks:
            score = pass_at_k_formula(n, c, k)
            aggregated_scores[k].append(score)

    final_scores = {k: sum(v)/len(v) for k,v in aggregated_scores.items()}
    return final_scores

def evaluate_all(data, max_workers=20):
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for item in data:
            model_ans = get_model_answer_after_think(item["response"][0]) 
            correct_ans = get_correct_answer_string(item)
            futures.append(executor.submit(gpt_score, model_ans, correct_ans))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Evaluating"):
            try:
                results.append(future.result())
            except Exception:
                results.append(0.0)

    return sum(results) / len(results) if results else 0.0

def final_accuracy(outputs):
    final_score = evaluate_all(outputs, max_workers=25)
    print(f"\n✅ Final Accuracy: {final_score:.3f}")

def final_accuracy_at_k(outputs, n):
    ks = [k for k in [1,2,4,8,16] if k <= n]
    results = evaluate_pass_at_k(outputs, ks, max_workers=25)
    for k, v in results.items():
        print(f"pass@{k}: {v:.3f}")