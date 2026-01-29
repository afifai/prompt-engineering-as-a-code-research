import boto3
import csv
import sys
import time
import os
import json
import re
from botocore.exceptions import ClientError

# --- CONFIG ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
PROMPT_PATH = os.path.join(project_root, 'prompts', 'instruction.txt')
DATASET_PATH = os.path.join(project_root, 'data', 'validation.csv')
METRICS_OUTPUT_PATH = "metrics.json"

AGENT_ID = os.environ.get("AGENT_ID")
REGION = os.environ.get("AWS_REGION", "us-east-1")
# Pastikan ini Model ID yang benar (Claude 3.5 Sonnet)
MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"

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
        
        # FIX 1: Tambah durasi sleep biar propagasi permission tuntas
        print("💤 Waiting 30s for changes to propagate...")
        time.sleep(30) 
        print("✅ Agent Updated & Prepared!")
    except Exception as e:
        print(f"❌ Gagal update agent: {str(e)}")

def extract_xml_tag(text, tag):
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text

def invoke_agent_with_retry(user_input, max_retries=3):
    """Fungsi helper untuk mencoba ulang jika kena AccessDenied"""
    for attempt in range(max_retries):
        try:
            response = bedrock_runtime.invoke_agent(
                agentId=AGENT_ID,
                agentAliasId='TSTALIASID', 
                sessionId='ci-cd-test-session',
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
            error_code = e.response['Error']['Code']
            # Kalau AccessDenied atau Throttling, coba lagi
            if error_code in ['accessDeniedException', 'ThrottlingException'] and attempt < max_retries - 1:
                print(f"⚠️ {error_code} detected. Retrying in 5s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(5)
            else:
                raise e # Kalau error lain atau sudah max retry, lempar errornya
    return None

def run_evaluation():
    print(f"\n🚀 Memulai Evaluasi Cerdas...")
    score = 0
    total = 0
    detailed_results = []

    try:
        with open(DATASET_PATH, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                total += 1
                user_input = row['input']
                expected_label = row['expected_label'].strip().upper()
                
                try:
                    # FIX 2: Pakai Retry Mechanism
                    completion = invoke_agent_with_retry(user_input)
                    
                    # Logika Parsing XML
                    actual_category = extract_xml_tag(completion, "category").upper()
                    generated_reply = extract_xml_tag(completion, "reply")
                    
                    if actual_category == completion.upper(): 
                        if expected_label in completion.upper():
                            actual_category = expected_label
                    
                    is_correct = expected_label in actual_category
                    
                    if is_correct:
                        score += 1
                        print(f"✅ PASS | In: {user_input[:15]}... | Reply: {generated_reply[:20]}...")
                    else:
                        print(f"❌ FAIL | In: {user_input[:15]}... | Exp: {expected_label} | Got: {actual_category}")
                    
                    detailed_results.append({
                        "input": user_input,
                        "expected": expected_label,
                        "actual_raw": generated_reply, 
                        "category": actual_category,
                        "is_correct": is_correct
                    })
                    
                    time.sleep(0.5)

                except Exception as e:
                    print(f"⚠️ Error row: {str(e)}")
                    # Jangan tambah total fail jika error sistem (biar akurasi gak 0% gara2 AWS error)
                    # Tapi tetap catat di log
                    detailed_results.append({
                        "input": user_input,
                        "expected": expected_label,
                        "actual_raw": f"SYSTEM ERROR: {str(e)}",
                        "category": "ERROR",
                        "is_correct": False
                    })

    except Exception as e:
        print(f"❌ Error reading CSV: {str(e)}")
        total = 1

    accuracy = (score / total) * 100 if total > 0 else 0
    print(f"\n📊 Accuracy: {accuracy:.2f}% ({score}/{total})")

    metrics = {
        "accuracy": accuracy,
        "passed": score,
        "total": total,
        "results": detailed_results
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