import boto3
import csv
import sys
import time
import os
import json
import uuid
from botocore.exceptions import ClientError

# --- CONFIG ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
PROMPT_PATH = os.path.join(project_root, 'prompts', 'instruction.txt')
DATASET_PATH = os.path.join(project_root, 'data', 'validation.csv')
METRICS_OUTPUT_PATH = "metrics.json"

AGENT_ID = os.environ.get("AGENT_ID")
REGION = os.environ.get("AWS_REGION", "us-east-1")
# Gunakan Claude 3 Sonnet (Legacy)
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
    # ... (BAGIAN INI TETAP SAMA SEPERTI SEBELUMNYA) ...
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

# --- DATA DEMO BARU (TANPA LABEL) ---
# Ini ceritanya data live yang masuk untuk dites reply-nya
DEMO_INPUTS = [
    "Mas, paket internet saya habis, ada promo murah ga?",
    "Selamat! Anda menang undian 500 juta, hubungi bapak budi.",
    "Pinjam dulu seratus besok diganti",
    "Kode OTP anda 4829. Jangan berikan ke siapapun.",
    "Kami dari kepolisian, anak anda tertangkap narkoba."
]

def run_evaluation():
    print(f"\n🚀 Memulai Evaluasi...")
    
    # 1. RUN VALIDATION (SAMA SEPERTI SEBELUMNYA)
    score = 0
    total = 0
    validation_results = []
    
    try:
        with open(DATASET_PATH, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                total += 1
                user_input = row['input']
                expected_label = row['expected_label'].strip().upper()
                
                completion = invoke_agent_with_retry(user_input)
                actual_raw = completion.strip()
                is_correct = expected_label in actual_raw.upper()
                
                if is_correct: score += 1
                
                validation_results.append({
                    "input": user_input,
                    "expected": expected_label,
                    "actual_raw": actual_raw,
                    "is_correct": is_correct
                })
                time.sleep(1) # Jeda dikit
    except Exception as e:
        print(f"❌ Error validation: {str(e)}")

    # 2. RUN DEMO INFERENCE (DATA BARU)
    print(f"\n🎮 Memulai Demo Inference (New Data)...")
    demo_results = []
    for text in DEMO_INPUTS:
        reply = invoke_agent_with_retry(text)
        print(f"📥 In: {text[:20]}... | 📤 Out: {reply[:20]}...")
        demo_results.append({
            "input": text,
            "reply": reply
        })
        time.sleep(1)

    # OUTPUT JSON
    accuracy = (score / total) * 100 if total > 0 else 0
    metrics = {
        "accuracy": accuracy,
        "passed": score,
        "total": total,
        "results": validation_results,
        "demo_results": demo_results # <--- INI DATA BARU
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