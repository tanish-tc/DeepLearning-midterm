import os
import pandas as pd
import re
import time
import xml.etree.ElementTree as ET
from together import Together
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# We use the specific 32B Coder Instruct model from Together AI
MODEL_NAME = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
TEST_CSV = "dataset/test.csv"
OUTPUT_CSV = "submission_together_qwen72b.csv"

# Concurrency: Keep this low (5-10) to avoid hitting Together API rate limits (HTTP 429)
MAX_WORKERS = 8 
MAX_RETRIES = 3

# Inference Parameters
TEMP = 0.2          # Keep it cold for strict syntax
TOP_P = 0.95
MAX_TOKENS = 4096
REP_PENALTY = 1.1

MAX_SVG_LENGTH = 8000
FALLBACK_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256"><rect width="256" height="256" fill="white"/><circle cx="128" cy="128" r="64" fill="black"/></svg>'
SVG_FULL_RE = re.compile(r"<svg[\s\S]*?</svg>", re.IGNORECASE)
ET.register_namespace("", "http://www.w3.org/2000/svg")

# Initialize the API Client
try:
    client = Together() # Defaults to os.environ.get("TOGETHER_API_KEY")
except Exception as e:
    print("❌ ERROR: Could not initialize Together client. Did you set TOGETHER_API_KEY?")
    exit()

SYSTEM_PROMPT = (
    "You are an expert SVG coder. Generate valid SVG markup from user requests. "
    "Return ONLY the SVG element with no explanation. "
    "Requirements: xmlns='http://www.w3.org/2000/svg', width='256', height='256', viewBox='0 0 256 256'."
)

# ==========================================
# 🛠️ VALIDATION & API CALL FUNCTIONS
# ==========================================
def extract_and_validate(text):
    """Aggressively extracts and strictly validates the SVG."""
    if not text.strip().endswith("</svg>"): text += "</svg>"
    
    # Strip markdown code blocks
    clean_text = text.replace("```xml", "").replace("```svg", "").replace("```", "").strip()
    
    m = SVG_FULL_RE.search(clean_text)
    svg = m.group(0).strip() if m else ""
    
    try:
        if svg and len(svg) <= MAX_SVG_LENGTH and ET.fromstring(svg).tag.endswith("svg"): 
            return svg
    except ET.ParseError: 
        pass
    return None

def fetch_svg_from_api(item_id, prompt):
    """Makes the API call with exponential backoff for rate limits."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMP,
                top_p=TOP_P,
                repetition_penalty=REP_PENALTY,
            )
            
            # Extract raw text from API response
            raw_text = response.choices[0].message.content
            
            # Validate it
            valid_svg = extract_and_validate(raw_text)
            if valid_svg:
                return {"id": item_id, "svg": valid_svg, "status": "success"}
            else:
                return {"id": item_id, "svg": FALLBACK_SVG, "status": "validation_failed"}
                
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate limit" in error_msg:
                # Exponential backoff for rate limits (2s, 4s, 8s)
                sleep_time = 2 ** (attempt + 1)
                time.sleep(sleep_time)
            else:
                # If it's a structural API error, fail instantly to fallback
                print(f"⚠️ API Error on ID {item_id}: {e}")
                break
                
    return {"id": item_id, "svg": FALLBACK_SVG, "status": "api_failed"}

# ==========================================
# ⚡ MULTI-THREADED GENERATION
# ==========================================
print(f"📂 Loading {TEST_CSV}...")
test_df = pd.read_csv(TEST_CSV)

print(f"🚀 Launching {MAX_WORKERS} concurrent workers to hit Together API...")
t0 = time.time()

submissions = []
stats = {"success": 0, "validation_failed": 0, "api_failed": 0}

# Using ThreadPoolExecutor to make concurrent network requests
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Submit all tasks to the queue
    future_to_id = {
        executor.submit(fetch_svg_from_api, row['id'], row['prompt']): row['id'] 
        for _, row in test_df.iterrows()
    }
    
    # Process them as they finish (with a progress bar)
    for future in tqdm(as_completed(future_to_id), total=len(test_df), desc="Generating via API"):
        result = future.result()
        submissions.append({"id": result["id"], "svg": result["svg"]})
        stats[result["status"]] += 1

# ==========================================
# 💾 SAVE SUBMISSION
# ==========================================
sub_df = pd.DataFrame(submissions)

# Sort the dataframe back into the original test.csv order
sub_df = sub_df.set_index('id').reindex(test_df['id']).reset_index()
sub_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n🎉 32B API Generation Complete in {(time.time()-t0)/60:.2f} mins.")
print("📊 Generation Stats:")
print(f"  - Perfect SVGs: {stats['success']}")
print(f"  - Fallbacks (Validation Failed - Too long/Broken XML): {stats['validation_failed']}")
print(f"  - Fallbacks (API/Rate Limit Failed): {stats['api_failed']}")
print(f"💾 Saved 32B output to {OUTPUT_CSV}")