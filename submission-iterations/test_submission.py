import pandas as pd
import json
import re

print("📂 Loading raw Kaggle training data...")
df = pd.read_csv("train.csv")

# The exact system prompt we want baked into the model's weights
SYSTEM_PROMPT = (
    "You are an expert SVG coder. Generate valid SVG markup from user requests. "
    "Return ONLY the SVG element with no explanation. "
    "Requirements: xmlns='http://www.w3.org/2000/svg', width='256', height='256', viewBox='0 0 256 256'."
)

clean_data = []
dropped_for_length = 0

for _, row in df.iterrows():
    prompt = str(row['prompt'])
    svg = str(row['svg'])
    
    # ==========================================
    # 🧼 THE DATA WASH
    # ==========================================
    # 1. Fix empty fills
    svg = re.sub(r'fill=""', 'fill="#000000"', svg)
    
    # 2. Remove the hallucinated 'filling' attribute
    svg = re.sub(r'\s+filling="[^"]*"', '', svg)
    
    # 3. Truncate absurd floats to 2 decimal places to save massive amounts of tokens
    svg = re.sub(r'(\d+\.\d{2})\d+', r'\1', svg)
    
    # 4. Enforce the strict Kaggle header 
    # (By training it on this header, it will never hallucinate 200x200 again)
    inner_content = re.sub(r'^<svg[^>]*>', '', svg, count=1)
    inner_content = re.sub(r'</svg>\s*$', '', inner_content)
    svg_final = f'<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">{inner_content}</svg>'
    
    # 5. The Compactness Filter: Drop anything over 7,000 characters
    if len(svg_final) > 7000:
        dropped_for_length += 1
        continue

    # ==========================================
    # 📦 TOGETHER AI FORMATTING
    # ==========================================
# ==========================================
    # 📦 TOGETHER AI FORMATTING (GEMMA SAFE)
    # ==========================================
    # We combine the system prompt into the user prompt for maximum Gemma compatibility
    combined_prompt = f"{SYSTEM_PROMPT}\n\nUser Request: {prompt}"
    
    conversation = {
        "messages": [
            {"role": "user", "content": combined_prompt},
            {"role": "assistant", "content": svg_final}
        ]
    }
    clean_data.append(conversation)

# Save to the Together AI required JSONL format
output_file = "together_train_washed.jsonl"
with open(output_file, 'w') as f:
    for item in clean_data:
        f.write(json.dumps(item) + '\n')

print(f"\n🎉 Wash and Conversion Complete!")
print(f"🗑️ Dropped {dropped_for_length} massive SVGs to maximize your compactness score.")
print(f"💾 Saved {len(clean_data)} mathematically perfect examples to {output_file}")