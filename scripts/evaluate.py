import boto3
import csv
import sys
import time
import os
import json

# --- UPDATE BAGIAN INI (DYNAMIC PATH FIX) ---
# Kita ambil lokasi script ini berada, lalu mundur satu folder ke belakang (root)
script_dir = os.path.dirname(os.path.abspath(__file__)) # Posisi: /.../repo/scripts
project_root = os.path.dirname(script_dir)              # Posisi: /.../repo

# Gabungkan path biar pasti ketemu dimanapun script dijalankan
PROMPT_PATH = os.path.join(project_root, 'prompts', 'instruction.txt')
DATASET_PATH = os.path.join(project_root, 'data', 'validation.csv')

# Debugging: Print path yang sedang dicoba akses (biar kelihatan di log)
print(f"📂 Project Root: {project_root}")
print(f"📄 Looking for Prompt at: {PROMPT_PATH}")
print(f"📄 Looking for Dataset at: {DATASET_PATH}")
print(f"📂 Files in Root: {os.listdir(project_root)}") # Cek isi folder root
# ----------------------------------------------

AGENT_ID = os.environ.get("AGENT_ID")
REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0" 
PASSING_SCORE = 80.0

# Init Boto3 Clients
bedrock_agent = boto3.client('bedrock-agent', region_name=REGION)
bedrock_runtime = boto3.client('bedrock-agent-runtime', region_name=REGION)

def get_agent_role_arn(agent_id):
    """Mengambil Role ARN agent secara otomatis biar gak perlu hardcode"""
    try:
        response = bedrock_agent.get_agent(agentId=agent_id)
        return response['agent']['agentResourceRoleArn']
    except Exception as e:
        print(f"❌ Error getting agent details: {str(e)}")
        sys.exit(1)

def update_and_prepare_agent(role_arn):
    """Update instruksi agent dan lakukan 'Prepare' (Compile)"""
    print(f"\n🔄 Membaca instruksi baru dari {PROMPT_PATH}...")
    try:
        with open(PROMPT_PATH, "r") as f:
            new_instruction = f.read()
    except FileNotFoundError:
        print("❌ File instruction.txt tidak ditemukan!")
        sys.exit(1)

    print("⚡ Meng-update Agent di AWS Bedrock...")
    try:
        bedrock_agent.update_agent(
            agentId=AGENT_ID,
            agentName='Spam-Detector-Agent', # Pastikan nama ini tidak konflik/berubah
            agentResourceRoleArn=role_arn,
            instruction=new_instruction,
            foundationModel=MODEL_ID
        )
        
        # Wajib Prepare agar perubahan aktif di Draft version
        print("⏳ Preparing Agent (Applying changes)...")
        bedrock_agent.prepare_agent(agentId=AGENT_ID)
        
        # Tunggu sebentar untuk propagasi (AWS best practice)
        time.sleep(10) 
        print("✅ Agent berhasil di-update dan siap dites!")
        
    except Exception as e:
        print(f"❌ Gagal update agent: {str(e)}")
        sys.exit(1)

def run_evaluation():
    """Jalankan tes menggunakan dataset CSV"""
    print(f"\n🚀 Memulai Evaluasi Otomatis (Quality Gate)...")
    
    score = 0
    total = 0
    failures = []

    try:
        with open(DATASET_PATH, 'r') as csvfile:
            # Menggunakan DictReader agar lebih aman baca kolom
            reader = csv.DictReader(csvfile)
            
            print(f"{'INPUT (Snippet)':<40} | {'EXPECTED':<10} | {'ACTUAL':<15} | {'RESULT'}")
            print("-" * 85)

            for row in reader:
                total += 1
                user_input = row['input']
                expected_label = row['expected_label'].strip().upper() # Normalize: SPAM/SAFE/OPERATORS
                
                # Invoke Agent (Hit ke TSTALIASID = Draft Version)
                try:
                    response = bedrock_runtime.invoke_agent(
                        agentId=AGENT_ID,
                        agentAliasId='TSTALIASID', 
                        sessionId='ci-cd-test-session', # Session ID statis utk tes
                        inputText=user_input,
                        enableTrace=False
                    )
                    
                    # Parsing Response Stream Bedrock
                    completion = ""
                    for event in response.get('completion'):
                        chunk = event['chunk']
                        if chunk:
                            completion += chunk['bytes'].decode('utf-8')
                    
                    # Logika Pengecekan (Case Insensitive & Substring Check)
                    # Agent mungkin jawab: "[SPAM] Karena ini penipuan..."
                    # Kita cek apakah kata "SPAM" ada di jawaban itu.
                    actual_response_upper = completion.upper()
                    
                    is_match = False
                    if expected_label in actual_response_upper:
                        is_match = True
                        score += 1
                        result_icon = "✅ PASS"
                    else:
                        result_icon = "❌ FAIL"
                        failures.append({
                            "input": user_input,
                            "expected": expected_label,
                            "got": completion
                        })

                    # Print baris tabel (potong input biar rapi)
                    print(f"{user_input[:37]+'...':<40} | {expected_label:<10} | {completion[:15]:<15} | {result_icon}")
                    
                    # Rate limit prevention (cegah throttling kalau datanya banyak)
                    time.sleep(1)

                except Exception as e:
                    print(f"⚠️ Error processing row: {str(e)}")
                    # Hitung error sebagai fail
                    total += 1

    except FileNotFoundError:
        print("❌ File validation.csv tidak ditemukan!")
        sys.exit(1)

    # --- HASIL AKHIR ---
    if total == 0:
        print("❌ Dataset kosong!")
        sys.exit(1)

    final_accuracy = (score / total) * 100
    print("\n" + "="*40)
    print(f"📊 FINAL REPORT")
    print(f"Total Test Cases : {total}")
    print(f"Passed           : {score}")
    print(f"Failed           : {total - score}")
    print(f"Accuracy         : {final_accuracy:.2f}%")
    print("="*40)

    # Cek Threshold Lulus/Gagal
    if final_accuracy >= PASSING_SCORE:
        print(f"🎉 SUCCESS: Score di atas {PASSING_SCORE}%. Pipeline Lulus.")
        sys.exit(0) # Exit Code 0 = GitHub Action Hijau
    else:
        print(f"⛔ FAILURE: Score di bawah {PASSING_SCORE}%. Pipeline Gagal.")
        print("\n🔍 Detail Kegagalan:")
        for fail in failures:
            print(f"- Input: {fail['input']}")
            print(f"  Expected: {fail['expected']} | Got: {fail['got']}\n")
        sys.exit(1) # Exit Code 1 = GitHub Action Merah

if __name__ == "__main__":
    if not AGENT_ID:
        print("❌ Environment Variable AGENT_ID belum diset!")
        sys.exit(1)
        
    # 1. Ambil Role ARN (Dynamic)
    role_arn = get_agent_role_arn(AGENT_ID)
    
    # 2. Update & Prepare Agent
    update_and_prepare_agent(role_arn)
    
    # 3. Jalankan Ujian
    run_evaluation()