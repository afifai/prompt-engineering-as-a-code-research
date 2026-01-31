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
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

bedrock_agent = boto3.client('bedrock-agent', region_name=REGION)
bedrock_runtime = boto3.client('bedrock-agent-runtime', region_name=REGION)

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
            agentName='Spam-Detector-Agent',
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
    Helper extract XML dengan 3 Layer Fallback
    """
    data = {
        "category": "UNKNOWN",
        "reason": "No reason provided",
        "reply": "NO_REPLY"
    }
    
    # --- LAYER 1: Standard XML Regex ---
    cat_match = re.search(r"<\s*category\s*>(.*?)<\s*/\s*category\s*>", text, re.DOTALL | re.IGNORECASE)
    reason_match = re.search(r"<\s*reason\s*>(.*?)<\s*/\s*reason\s*>", text, re.DOTALL | re.IGNORECASE)
    reply_match = re.search(r"<\s*reply\s*>(.*?)<\s*/\s*reply\s*>", text, re.DOTALL | re.IGNORECASE)

    if cat_match:
        data["category"] = cat_match.group(1).replace("[", "").replace("]", "").strip().upper()
    if reason_match:
        data["reason"] = reason_match.group(1).strip()
    if reply_match:
        data["reply"] = reply_match.group(1).strip()

    # --- LAYER 2: Fallback (Cari Bracket [CATEGORY] jika XML gagal) ---
    if data["category"] == "UNKNOWN":
        if "[SPAM]" in text.upper(): data["category"] = "SPAM"
        elif "[OPERATORS]" in text.upper(): data["category"] = "OPERATORS"
        elif "[SAFE]" in text.upper(): data["category"] = "SAFE"
    
    # --- LAYER 3: Fallback Extreme (Cari kata kunci polos di awal teks) ---
    if data["category"] == "UNKNOWN":
        upper_text = text.upper()
        # Asumsi AI nulis: "Category: SPAM"
        if "CATEGORY: SPAM" in upper_text: data["category"] = "SPAM"
        elif "CATEGORY: OPERATORS" in upper_text: data["category"] = "OPERATORS"
        elif "CATEGORY: SAFE" in upper_text: data["category"] = "SAFE"

    # --- FINAL CLEANUP ---
    # Pastikan category valid, kalau tidak paksa ke SAFE (Fail-safe)
    valid_cats = ["SPAM", "OPERATORS", "SAFE"]
    if data["category"] not in valid_cats:
        # Jika text pendek/gantung biasanya SAFE
        data["category"] = "SAFE" 
    
    return data

def run_evaluation():
    print(f"\n🚀 Memulai Evaluasi...")
    
    # 1. VALIDATION
    score = 0
    total = 0
    validation_results = []
    
    print(f"📂 Reading Validation Data...")
    try:
        with open(VALIDATION_PATH, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                total += 1
                user_input = row['input']
                expected_label = row['expected_label'].strip().upper()
                
                raw_response = invoke_agent_with_retry(user_input)
                parsed = extract_xml_data(raw_response)
                
                # Check correctness
                # Kita cek substring karena AI mungkin jawab "SPAM" atau "[SPAM]"
                is_correct = expected_label in parsed["category"]
                
                if is_correct: score += 1
                
                validation_results.append({
                    "input": user_input,
                    "expected": expected_label,
                    "actual_category": parsed["category"], # Label Prediksi
                    "is_correct": is_correct
                })
                time.sleep(1)
    except Exception as e:
        print(f"❌ Error validation: {str(e)}")

    # 2. INFERENCE
    print(f"\n📂 Reading Inference Data...")
    demo_results = []
    try:
        with open(INFERENCE_PATH, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                user_input = row['input']
                raw_response = invoke_agent_with_retry(user_input)
                parsed = extract_xml_data(raw_response)
                
                print(f"📥 In: {user_input[:15]}... | 🏷️ Cat: {parsed['category']}")
                
                demo_results.append({
                    "input": user_input,
                    "category": parsed["category"],
                    "reason": parsed["reason"],
                    "reply": parsed["reply"]
                })
                time.sleep(1)
    except Exception as e:
         print(f"❌ Error inference: {str(e)}")

    accuracy = (score / total) * 100 if total > 0 else 0
    metrics = {
        "accuracy": accuracy,
        "passed": score,
        "total": total,
        "results": validation_results,
        "demo_results": demo_results
    }
    
    with open(METRICS_OUTPUT_PATH, "w") as f:
        json.dump(metrics, f)
    
    sys.exit(0)

if __name__ == "__main__":
    if AGENT_ID:
        role_arn = get_agent_role_arn(AGENT_ID)
        if role_arn:
            update_and_prepare_agent(role_arn)
            run_evaluation()
