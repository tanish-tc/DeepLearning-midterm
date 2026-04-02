import os
import time
import re
import pandas as pd
import xml.etree.ElementTree as ET

# 1. BULLETPROOF ENVIRONMENT SETUP
# Force stable V0 engine and restrict to a SINGLE GPU to prevent all NCCL/Socket crashes
os.environ["VLLM_USE_V1"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

from vllm import LLM, SamplingParams

# 2. PATHS & CONFIG
MERGED_DIR    = "./merged_7B"   # Update to your actual merged model folder!
TEST_CSV      = "./test.csv"
SUBMISSION    = "./submission_final.csv"

SYSTEM_PROMPT = (
    "You generate valid SVG markup from user requests. "
    "Return ONLY the SVG element with no explanation. "
    "Requirements: xmlns='http://www.w3.org/2000/svg', "
    "width='256', height='256', viewBox='0 0 256 256'. "
    "Use only standard SVG attributes."
)

# 3. BOOT vLLM (Single A100)
print(f"\n🚀 Booting vLLM Engine on a Single A100...")
llm = LLM(
    model                  = MERGED_DIR,
    max_model_len          = 3072,
    tensor_parallel_size   = 1,      # Absolutely no multi-GPU networking bugs!
    gpu_memory_utilization = 0.90,   
    enforce_eager          = True,   
    disable_log_stats      = False,
)
print("✅ vLLM Engine ready!")

# 4. THE GOLDILOCKS PARAMS (Learned from the Mac)
sampling_params = SamplingParams(
    temperature=0.3,           # Enough entropy to prevent coordinate loops
    presence_penalty=0.5,      # Taxes infinite loops without breaking valid SVG syntax
    max_tokens=2500,
    stop=["</svg>", "<|im_end|>", "<|endoftext|>"] # Instant early stopping
)

# 5. RUN INFERENCE
print(f"\n📂 Reading {TEST_CSV}...")
test_df = pd.read_csv(TEST_CSV)
prompts = test_df["prompt"].tolist()
ids     = test_df["id"].tolist()

# MUST use hardcoded ChatML!
chat_texts = [
    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    f"<|im_start|>user\n{p}<|im_end|>\n"
    f"<|im_start|>assistant\n"
    for p in prompts
]

print(f"⚡ Feeding {len(chat_texts)} prompts into vLLM batch engine. Hold on tight...")
t0 = time.time()
outputs = llm.generate(chat_texts, sampling_params)

# 6. POST-PROCESSING & XML VALIDATION
ET.register_namespace("", "http://www.w3.org/2000/svg")
SVG_FULL_RE = re.compile(r"<svg[\s\S]*?</svg>", re.IGNORECASE)

def extract_svg(text):
    m = SVG_FULL_RE.search(text)
    return m.group(0).strip() if m else ""

def is_valid_svg(svg_text):
    if not svg_text: return False
    try:
        return ET.fromstring(svg_text).tag.endswith("svg")
    except ET.ParseError:
        return False

def fallback_svg():
    return '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256"><rect width="256" height="256" fill="white"/><circle cx="128" cy="128" r="64" fill="black"/></svg>'

rows = []
invalid_count = 0

print("\n🛠️ Validating XML...")
for i, output in enumerate(outputs):
    generated_text = output.outputs[0].text
    
    # Re-add the closing tag because our early-stopping feature strips it out!
    if not generated_text.strip().endswith("</svg>"):
        generated_text += "</svg>"

    svg = extract_svg(generated_text)
    
    if is_valid_svg(svg):
        final_svg = svg
    else:
        invalid_count += 1
        final_svg = fallback_svg()
        
    rows.append({"id": ids[i], "svg": final_svg})

# 7. SAVE RESULTS
os.makedirs(os.path.dirname(SUBMISSION), exist_ok=True)
pd.DataFrame(rows).to_csv(SUBMISSION, index=False)

final_time = (time.time() - t0) / 60
print(f"\n🎉 Blazing fast inference complete! Saved to {SUBMISSION}")
print(f"⚠️ Fallback count: {invalid_count} / {len(prompts)}")
print(f"⏱️ Total Batched Runtime: {final_time:.2f} minutes\n")