import boto3
import csv
import sys
import time
import os
import json
import uuid  # <--- Upgrade: Biar Session ID unik
from botocore.exceptions import ClientError

# --- CONFIG ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
PROMPT_PATH = os.path.join(project_root, 'prompts', 'instruction.txt')
DATASET_PATH = os.path.join(project_root, 'data', 'validation.csv')
METRICS_OUTPUT_PATH = "metrics.json"

AGENT_ID = os.environ.get("AGENT_ID")
REGION = os.environ.get("AWS_REGION", "us-east-1")
# Pastikan ID Model Benar (Claude 3.5 Sonnet)
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

    # --- TAMBAHAN DEBUGGING ---
    print(f"🕵️  Agent menggunakan Role: {role_arn}") 
    print("    (Pastikan Role di atas punya izin 'AmazonBedrockFullAccess'!)")
    # --------------------------

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
    """
    Mencoba invoke agent dengan:
    1. Session ID Unik (Biar gak locking)
    2. Retry Delay Panjang (Biar gak kena rate limit)
    """
    # Gunakan Session ID unik setiap request
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
            error_code = e.response['Error']['Code']
            # AccessDenied seringkali sebenarnya adalah Throttling di Bedrock Agent
            if error_code in ['accessDeniedException', 'ThrottlingException'] and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5 + 5 # 10s, 15s, 20s
                print(f"⚠️ {error_code} detected. Cooling down {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e 
    return None

def run_evaluation():
    print(f"\n🚀 Memulai Evaluasi (Mode Stabil)...")
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
                    # Panggil fungsi retry
                    completion = invoke_agent_with_retry(user_input)
                    
                    if completion:
                        actual_raw = completion.strip()
                        # Cek simple string matching (tanpa XML)
                        is_correct = expected_label in actual_raw.upper()
                        
                        if is_correct:
                            score += 1
                            print(f"✅ PASS | In: {user_input[:15]}... | Out: {actual_raw[:20]}...")
                        else:
                            print(f"❌ FAIL | In: {user_input[:15]}... | Exp: {expected_label} | Got: {actual_raw[:20]}...")
                        
                        detailed_results.append({
                            "input": user_input,
                            "expected": expected_label,
                            "actual_raw": actual_raw,
                            "is_correct": is_correct
                        })
                    else:
                        raise Exception("Empty response after retries")

                    # Jeda antar request biar Agent gak pusing (Throttling Prevention)
                    time.sleep(2) 

                except Exception as e:
                    print(f"⚠️ Error row: {str(e)}")
                    # Tetap catat error biar report PR lengkap
                    detailed_results.append({
                        "input": user_input,
                        "expected": expected_label,
                        "actual_raw": f"ERROR: {str(e)}",
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