import boto3
import csv
import sys
import time
import os
import json
import uuid
import re
from botocore.exceptions import ClientError

# --- CONFIG ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
PROMPT_PATH = os.path.join(project_root, 'prompts', 'instruction.txt')
VALIDATION_PATH = os.path.join(project_root, 'data', 'validation.csv')
INFERENCE_PATH = os.path.join(project_root, 'data', 'inference.csv')
METRICS_OUTPUT_PATH = "metrics.json"

AGENT_ID = os.environ.get("AGENT_ID")
REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
JUDGE_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

bedrock_agent = boto3.client('bedrock-agent', region_name=REGION)
bedrock_runtime = boto3.client('bedrock-agent-runtime', region_name=REGION)
bedrock_invoke = boto3.client('bedrock-runtime', region_name=REGION)


def get_agent_role_arn(agent_id):
    try:
        response = bedrock_agent.get_agent(agentId=agent_id)
        return response['agent']['agentResourceRoleArn']
    except Exception as e:
        print(f"❌ Error getting agent details: {str(e)}")
        return None


def update_and_prepare_agent(role_arn):
    print(f"\n🔄 Membaca instruksi dari {PROMPT_PATH}...")
    try:
        with open(PROMPT_PATH, "r") as f:
            new_instruction = f.read()
    except FileNotFoundError:
        print("❌ File instruction.txt tidak ditemukan!")
        return

    print("⚡ Meng-update Agent di AWS Bedrock...")
    try:
        bedrock_agent.update_agent(
            agentId=AGENT_ID,
            agentName='Translation-Agent',
            agentResourceRoleArn=role_arn,
            instruction=new_instruction,
            foundationModel=MODEL_ID
        )
        print("⏳ Preparing Agent (Applying changes)...")
        bedrock_agent.prepare_agent(agentId=AGENT_ID)
        print("💤 Waiting 30s for Agent propagation...")
        time.sleep(30)
        print("✅ Agent Updated & Prepared!")
    except Exception as e:
        print(f"❌ Gagal update agent: {str(e)}")


def invoke_agent_with_retry(user_input, max_retries=3):
    session_id = str(uuid.uuid4())
    for attempt in range(max_retries):
        try:
            response = bedrock_runtime.invoke_agent(
                agentId=AGENT_ID,
                agentAliasId='TSTALIASID',
                sessionId=session_id,
                inputText=user_input,
                enableTrace=False
            )
            completion = ""
            for event in response.get('completion'):
                chunk = event['chunk']
                if chunk:
                    completion += chunk['bytes'].decode('utf-8')
            return completion
        except ClientError as e:
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return f"SYSTEM_ERROR: {str(e)}"
    return "SYSTEM_ERROR: Max retries reached"


def extract_xml_data(text):
    """
    Extract XML fields from the translator agent's response.
    """
    data = {
        "result": "",
        "difficulty": "UNKNOWN",
        "analysis": "No analysis provided",
        "notes": "NONE"
    }

    result_match = re.search(
        r"<\s*result\s*>(.*?)<\s*/\s*result\s*>",
        text, re.DOTALL | re.IGNORECASE
    )
    if result_match:
        data["result"] = result_match.group(1).strip()

    diff_match = re.search(
        r"<\s*difficulty\s*>(.*?)<\s*/\s*difficulty\s*>",
        text, re.DOTALL | re.IGNORECASE
    )
    if diff_match:
        clean = diff_match.group(1).replace("[", "").replace("]", "").strip().upper()
        data["difficulty"] = clean

    analysis_match = re.search(
        r"<\s*analysis\s*>(.*?)<\s*/\s*analysis\s*>",
        text, re.DOTALL | re.IGNORECASE
    )
    if analysis_match:
        data["analysis"] = analysis_match.group(1).strip()

    notes_match = re.search(
        r"<\s*notes\s*>(.*?)<\s*/\s*notes\s*>",
        text, re.DOTALL | re.IGNORECASE
    )
    if notes_match:
        data["notes"] = notes_match.group(1).strip()

    return data


# ============================================================
# LLM-as-a-Judge: Uses a separate model to evaluate translation
# quality instead of simple string matching.
# ============================================================

JUDGE_PROMPT_TEMPLATE = """You are an expert bilingual judge evaluating Indonesian-to-English translations.

Given:
- Original Indonesian text
- Expected (reference) English translation
- Candidate English translation produced by an AI

Score the candidate translation on a scale of 1-5:
  5 = Perfect or near-perfect. Meaning, tone, and fluency are all preserved.
  4 = Good. Minor stylistic differences but meaning is fully preserved.
  3 = Acceptable. Meaning is mostly correct but some nuance is lost or phrasing is awkward.
  2 = Poor. Significant meaning errors or very unnatural phrasing.
  1 = Wrong. Major meaning errors, nonsensical, or not a translation at all.

Also evaluate the difficulty classification:
- Expected difficulty: {expected_difficulty}
- Predicted difficulty: {predicted_difficulty}
- Is the difficulty classification correct? (true/false)

Respond ONLY in this exact XML format:

<judge>
    <score>NUMBER</score>
    <difficulty_correct>true or false</difficulty_correct>
    <reasoning>Brief explanation of your scoring decision in English.</reasoning>
</judge>

---
Original (Indonesian): {original}
Expected Translation: {expected}
Candidate Translation: {candidate}
"""


def invoke_judge(original, expected, candidate, expected_difficulty, predicted_difficulty, max_retries=3):
    """
    Call a separate LLM (Haiku) to judge translation quality.
    This is the LLM-as-a-Judge pattern.
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        original=original,
        expected=expected,
        candidate=candidate,
        expected_difficulty=expected_difficulty,
        predicted_difficulty=predicted_difficulty
    )

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    })

    for attempt in range(max_retries):
        try:
            response = bedrock_invoke.invoke_model(
                modelId=JUDGE_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=body
            )
            result = json.loads(response['body'].read())
            text = result['content'][0]['text']
            return parse_judge_response(text)
        except ClientError as e:
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                print(f"❌ Judge error: {str(e)}")
                return {"score": 0, "difficulty_correct": False, "reasoning": f"JUDGE_ERROR: {str(e)}"}

    return {"score": 0, "difficulty_correct": False, "reasoning": "JUDGE_ERROR: Max retries reached"}


def parse_judge_response(text):
    """Parse the judge model's XML response."""
    data = {
        "score": 0,
        "difficulty_correct": False,
        "reasoning": "No reasoning provided"
    }

    score_match = re.search(
        r"<\s*score\s*>(.*?)<\s*/\s*score\s*>",
        text, re.DOTALL | re.IGNORECASE
    )
    if score_match:
        try:
            data["score"] = int(score_match.group(1).strip())
        except ValueError:
            data["score"] = 0

    diff_match = re.search(
        r"<\s*difficulty_correct\s*>(.*?)<\s*/\s*difficulty_correct\s*>",
        text, re.DOTALL | re.IGNORECASE
    )
    if diff_match:
        data["difficulty_correct"] = diff_match.group(1).strip().lower() == "true"

    reason_match = re.search(
        r"<\s*reasoning\s*>(.*?)<\s*/\s*reasoning\s*>",
        text, re.DOTALL | re.IGNORECASE
    )
    if reason_match:
        data["reasoning"] = reason_match.group(1).strip()

    return data


def run_evaluation():
    print(f"\n🚀 Memulai Evaluasi...")

    # ========================================
    # 1. VALIDATION (with LLM-as-a-Judge)
    # ========================================
    total_score = 0
    total = 0
    difficulty_correct_count = 0
    validation_results = []

    print(f"📂 Reading Validation Data...")
    try:
        with open(VALIDATION_PATH, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                total += 1
                user_input = row['input']
                expected_translation = row['expected_translation'].strip()
                expected_difficulty = row['expected_difficulty'].strip().upper()

                # Step 1: Get translation from the Agent (with retry for empty results)
                agent_input = f"Terjemahkan kalimat berikut ke Bahasa Inggris:\n\n{user_input}"
                parsed = None
                for translate_attempt in range(3):
                    raw_response = invoke_agent_with_retry(agent_input)
                    print(f"    📝 Raw ({translate_attempt+1}): {raw_response[:100]}...")
                    parsed = extract_xml_data(raw_response)
                    if parsed["result"] and "SYSTEM_ERROR" not in parsed["result"] and "asisten AI" not in parsed["result"]:
                        break
                    print(f"    ⚠️  Empty/bad translation, retrying ({translate_attempt + 1}/3)...")
                    time.sleep(3)

                # Step 2: Use LLM-as-a-Judge to evaluate quality
                judge_result = invoke_judge(
                    original=user_input,
                    expected=expected_translation,
                    candidate=parsed["result"],
                    expected_difficulty=expected_difficulty,
                    predicted_difficulty=parsed["difficulty"]
                )

                if judge_result["difficulty_correct"]:
                    difficulty_correct_count += 1

                total_score += judge_result["score"]

                validation_results.append({
                    "input": user_input,
                    "expected_translation": expected_translation,
                    "actual_translation": parsed["result"],
                    "expected_difficulty": expected_difficulty,
                    "actual_difficulty": parsed["difficulty"],
                    "judge_score": judge_result["score"],
                    "difficulty_correct": judge_result["difficulty_correct"],
                    "judge_reasoning": judge_result["reasoning"],
                    "analysis": parsed["analysis"]
                })

                print(f"  📥 In: {user_input[:30]}... | 🏷️ Score: {judge_result['score']}/5 | Diff: {parsed['difficulty']}")
                time.sleep(1)
    except Exception as e:
        print(f"❌ Error validation: {str(e)}")

    # ========================================
    # 2. INFERENCE (generative showcase)
    # ========================================
    print(f"\n📂 Reading Inference Data...")
    demo_results = []
    try:
        with open(INFERENCE_PATH, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                user_input = row['input']

                # Retry for empty results
                agent_input = f"Terjemahkan kalimat berikut ke Bahasa Inggris:\n\n{user_input}"
                parsed = None
                for translate_attempt in range(3):
                    raw_response = invoke_agent_with_retry(agent_input)
                    parsed = extract_xml_data(raw_response)
                    if parsed["result"] and "SYSTEM_ERROR" not in parsed["result"] and "asisten AI" not in parsed["result"]:
                        break
                    print(f"    ⚠️  Empty/bad translation, retrying ({translate_attempt + 1}/3)...")
                    time.sleep(3)

                print(f"  📥 In: {user_input[:30]}... | 🏷️ Diff: {parsed['difficulty']}")

                demo_results.append({
                    "input": user_input,
                    "translation": parsed["result"],
                    "difficulty": parsed["difficulty"],
                    "analysis": parsed["analysis"],
                    "notes": parsed["notes"]
                })
                time.sleep(1)
    except Exception as e:
        print(f"❌ Error inference: {str(e)}")

    # ========================================
    # 3. Compute Metrics & Save
    # ========================================
    avg_score = (total_score / total) if total > 0 else 0
    max_possible = total * 5
    quality_pct = (total_score / max_possible) * 100 if max_possible > 0 else 0
    difficulty_accuracy = (difficulty_correct_count / total) * 100 if total > 0 else 0

    metrics = {
        "avg_judge_score": round(avg_score, 2),
        "quality_percentage": round(quality_pct, 2),
        "difficulty_accuracy": round(difficulty_accuracy, 2),
        "total_score": total_score,
        "max_score": max_possible,
        "total": total,
        "difficulty_correct": difficulty_correct_count,
        "results": validation_results,
        "demo_results": demo_results
    }

    with open(METRICS_OUTPUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n📊 === RESULTS ===")
    print(f"  Avg Judge Score : {avg_score:.2f} / 5")
    print(f"  Quality         : {quality_pct:.2f}%")
    print(f"  Difficulty Acc  : {difficulty_accuracy:.2f}%")
    print(f"  Total Evaluated : {total}")

    sys.exit(0)


if __name__ == "__main__":
    if AGENT_ID:
        role_arn = get_agent_role_arn(AGENT_ID)
        if role_arn:
            update_and_prepare_agent(role_arn)
            run_evaluation()
