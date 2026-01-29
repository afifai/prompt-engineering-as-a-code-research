import boto3
import csv
import sys
import time
import os
import json

# --- CONFIG ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
PROMPT_PATH = os.path.join(project_root, 'prompts', 'instruction.txt')
DATASET_PATH = os.path.join(project_root, 'data', 'validation.csv')
METRICS_OUTPUT_PATH = "metrics.json"

AGENT_ID = os.environ.get("AGENT_ID")
REGION = os.environ.get("AWS_REGION", "us-east-1")
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
        time.sleep(10) 
        print("✅ Agent Updated & Prepared!")
    except Exception as e:
        print(f"❌ Gagal update agent: {str(e)}")

def run_evaluation():
    print(f"\n🚀 Memulai Evaluasi...")
    score = 0
    total = 0
    detailed_results = [] # <--- KITA SIMPAN SEMUA HASIL DI SINI

    try:
        with open(DATASET_PATH, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                total += 1
                user_input = row['input']
                expected_label = row['expected_label'].strip().upper()
                
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
                    
                    actual_upper = completion.upper()
                    is_correct = expected_label in actual_upper
                    
                    if is_correct:
                        score += 1
                        print(f"✅ PASS | In: {user_input[:20]}...")
                    else:
                        print(f"❌ FAIL | In: {user_input[:20]}... | Exp: {expected_label} | Got: {completion[:20]}...")
                    
                    # Simpan detail untuk laporan perbandingan
                    detailed_results.append({
                        "input": user_input,
                        "expected": expected_label,
                        "actual_raw": completion, # Jawaban lengkap agent
                        "is_correct": is_correct
                    })
                    
                    time.sleep(0.5)

                except Exception as e:
                    print(f"⚠️ Error row: {str(e)}")
                    total += 1
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

    # --- SIMPAN JSON LENGKAP ---
    metrics = {
        "accuracy": accuracy,
        "passed": score,
        "total": total,
        "results": detailed_results # List ini akan dibaca oleh GitHub Action
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