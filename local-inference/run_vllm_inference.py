import os
import re
import time
import pandas as pd
import xml.etree.ElementTree as ET
from mlx_lm import load, generate

# 1. PATHS
MODEL_PATH = "./merged_model"  # Point this to your unzipped folder
TEST_CSV   = "./test.csv"
SUBMISSION = "./submission_final.csv"

SYSTEM_PROMPT = (
    "You generate valid SVG markup from user requests. "
    "Return ONLY the SVG element with no explanation. "
    "Requirements: xmlns='http://www.w3.org/2000/svg', "
    "width='256', height='256', viewBox='0 0 256 256'. "
    "Use only standard SVG attributes."
)

# 2. LOAD MODEL INTO APPLE SILICON (M4)
print(f"\n🚀 Booting MLX Engine on Apple M4...")
model, tokenizer = load(MODEL_PATH)
print("✅ MLX Engine ready!")

# 3. HELPER FUNCTIONS
ET.register_namespace("", "http://www.w3.org/2000/svg")
SVG_FULL_RE = re.compile(r"<svg[\s\S]*?</svg>", re.IGNORECASE)

def extract_svg(text):
    m = SVG_FULL_RE.search(text)
    return m.group(0).strip() if m else ""

def is_valid_svg(svg):
    if not svg: return False
    try: return ET.fromstring(svg).tag.split("}")[-1] == "svg"
    except ET.ParseError: return False

def fallback_svg():
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">'
            '<rect width="256" height="256" fill="white"/>'
            '<circle cx="128" cy="128" r="64" fill="black"/></svg>')

# 4. RUN INFERENCE
print(f"📂 Reading {TEST_CSV}...")
test_df = pd.read_csv(TEST_CSV)
prompts = test_df["prompt"].tolist()
ids     = test_df["id"].tolist()

rows = []
invalid_count = 0
t0 = time.time()

print(f"⚡ Generating {len(prompts)} SVGs on Mac M4...")
for i, p in enumerate(prompts):
    # Format the prompt
    prompt_text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{p}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    # Generate using Apple MLX
    response = generate(
        model, 
        tokenizer, 
        prompt=prompt_text, 
        max_tokens=2048, 
        verbose=False
    )
    
    # Process
    svg = extract_svg(response)
    if is_valid_svg(svg):
        final_svg = svg
    else:
        invalid_count += 1
        final_svg = fallback_svg()
        
    rows.append({"id": ids[i], "svg": final_svg})
    
    # Print progress every 10 items
    if (i + 1) % 10 == 0:
        print(f"Progress: {i + 1} / {len(prompts)} done...")

# 5. SAVE
pd.DataFrame(rows).to_csv(SUBMISSION, index=False)

elapsed = (time.time() - t0) / 60
print(f"\n🎉 Generation complete in {elapsed:.2f} minutes!")
print(f"⚠️ Fallbacks used: {invalid_count} / {len(prompts)}")
print(f"✅ Saved to {SUBMISSION}")