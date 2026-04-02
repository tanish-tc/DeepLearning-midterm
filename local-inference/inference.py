#!/usr/bin/env python3
"""
Text-to-SVG Inference — M4 Mac Pro 24GB
Uses mlx-lm: Apple's native inference engine, fastest on M-series
"""

import re, time, xml.etree.ElementTree as ET
import pandas as pd
from mlx_lm import load, generate

# ============================================================
# PATHS
# ============================================================
BASE_MODEL_DIR   = "unsloth/Ministral-3-3B-Instruct-2512-unsloth-bnb-4bit"
LORA_ADAPTER_DIR = "./qwen_svg_lora_3B"
TEST_CSV         = "./test.csv"
SUBMISSION_CSV   = "./submission_mac.csv"

SYSTEM_PROMPT = (
    "You generate valid SVG markup from user requests. "
    "Return ONLY the SVG element with no explanation. "
    "Requirements: xmlns='http://www.w3.org/2000/svg', "
    "width='256', height='256', viewBox='0 0 256 256'. "
    "Use only standard SVG attributes."
)

# ============================================================
# LOAD MODEL
# mlx-lm handles the Mistral3 architecture natively
# and automatically uses Metal GPU on Apple Silicon
# ============================================================
print("Loading model with mlx-lm (first run downloads ~6GB)...")
model, tokenizer = load(
    BASE_MODEL_DIR,
    adapter_path=LORA_ADAPTER_DIR,   # applies LoRA automatically
)
print("Model ready.\n")

# ============================================================
# POST-PROCESSING (unchanged)
# ============================================================
ET.register_namespace("", "http://www.w3.org/2000/svg")
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
SVG_FULL_RE = re.compile(r"<svg[\s\S]*?</svg>", re.IGNORECASE)


def truncate_at_loop(text):
    m = re.search(r"(.{30,}?)\1{3,}", text)
    if m:
        return text[: m.start() + len(m.group(1))], True
    return text, False


def extract_svg(text):
    m = SVG_FULL_RE.search(text)
    return m.group(0).strip() if m else ""


def repair_svg(text):
    start = text.lower().find("<svg")
    if start == -1:
        return None
    text = text[start:].strip()
    if re.search(r"</svg\s*>", text, re.IGNORECASE):
        return text
    boundary = max(text.rfind("/>"), text.rfind(">"))
    if boundary == -1:
        return None
    text = text[:boundary + 1]
    VOID = {"circle","ellipse","line","path","polygon",
            "polyline","rect","use","image","stop"}
    open_tags = []
    for m in re.finditer(r"<(/?)([\w:_-]+)([^>]*?)(/?)>", text):
        is_close = m.group(1) == "/"
        tag      = m.group(2).lower()
        sc       = m.group(4) == "/"
        if tag == "svg":
            if not is_close:
                open_tags = []
            continue
        if sc or tag in VOID:
            continue
        if is_close:
            if open_tags and open_tags[-1] == tag:
                open_tags.pop()
        else:
            open_tags.append(tag)
    return text + "".join(f"</{t}>" for t in reversed(open_tags)) + "</svg>"


def is_valid_svg(svg):
    if not svg:
        return False
    try:
        return ET.fromstring(svg).tag.split("}")[-1] == "svg"
    except ET.ParseError:
        return False


def deduplicate_svg(svg):
    try:
        root = ET.fromstring(svg)
    except ET.ParseError:
        return svg, 0
    children = list(root)
    if not children:
        return svg, 0
    seen, to_remove = set(), []
    for child in children:
        tag = child.tag.split("}")[-1]
        d   = child.get("d", "")
        if tag == "path" and d:
            coords = re.findall(r"-?\d+\.?\d*", d)
            if len(coords) > 2 and len(set(coords)) <= 2:
                to_remove.append(child)
                continue
        key = (child.tag, tuple(sorted(child.attrib.items())))
        if key in seen:
            to_remove.append(child)
        else:
            seen.add(key)
    for c in to_remove[:max(0, len(children) - 1)]:
        root.remove(c)
    return ET.tostring(root, encoding="unicode"), len(to_remove)


def clean_and_normalize(svg):
    svg = re.sub(r'\s*filling="[^"]*"', "", svg)
    svg = re.sub(r'fill=""', 'fill="#000000"', svg)
    svg = re.sub(r"\s*fill-opacity=\"1\"", "", svg)
    svg = re.sub(r'viewBox=["\'][\s\d]+["\']', 'viewBox="0 0 256 256"', svg)
    svg = re.sub(r'(?<![.\w])height=["\']\d+(?:px)?["\']', 'height="256"', svg)
    svg = re.sub(r'(?<![.\w])width=["\']\d+(?:px)?["\']',  'width="256"',  svg)
    svg = re.sub(r"[\n\r\t]", "", svg)
    svg = re.sub(r">\s+<", "><", svg)
    svg = re.sub(r"\s*=\s*", "=", svg)
    svg = re.sub(r"\s+", " ", svg)
    def _round(m):
        v = float(m.group(0))
        return str(int(v)) if v == int(v) else f"{v:.1f}"
    return re.sub(r"-?\d+\.\d+", _round, svg).strip()


def fallback_svg():
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="256" height="256" viewBox="0 0 256 256">'
        '<rect width="256" height="256" fill="white"/>'
        '<circle cx="128" cy="128" r="64" fill="black"/>'
        "</svg>"
    )


def process_output(text, prompt=""):
    cleaned, was_looped = truncate_at_loop(text)
    svg = extract_svg(cleaned)
    if not is_valid_svg(svg):
        svg = repair_svg(cleaned)
        if not svg or not is_valid_svg(svg):
            return fallback_svg(), "fallback"
        stage = "loop_repaired" if was_looped else "repaired"
    else:
        stage = "loop_direct" if was_looped else "direct"
    svg, n = deduplicate_svg(svg)
    if not svg:
        return fallback_svg(), "fallback"
    if n > 0:
        stage += "_deduped"
    svg = clean_and_normalize(svg)
    if not is_valid_svg(svg):
        return fallback_svg(), "fallback_post_clean"
    return svg, stage


# ============================================================
# INFERENCE
# mlx-lm runs one prompt at a time but is fast enough on M4
# ============================================================
print(f"Reading {TEST_CSV}...")
test_df = pd.read_csv(TEST_CSV)
prompts = test_df["prompt"].tolist()
ids     = test_df["id"].tolist()
print(f"Prompts: {len(prompts):,}\n")


def format_prompt(p: str) -> str:
    # Try the tokenizer's built-in chat template first
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": p},
            ]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    # Fallback: ChatML format (Unsloth default)
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{p}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


all_generated = []
t0 = time.time()

for i, prompt in enumerate(prompts):
    formatted = format_prompt(prompt)

    response = generate(
        model,
        tokenizer,
        prompt=formatted,
        max_tokens=1800,
        verbose=False,
    )
    all_generated.append(response)

    if (i + 1) % 10 == 0:
        elapsed = time.time() - t0
        eta     = (len(prompts) - (i+1)) / ((i+1) / elapsed)
        print(
            f"  [{i+1:>5}/{len(prompts)}]  "
            f"{elapsed/60:.1f}min elapsed  "
            f"ETA {eta/60:.1f}min",
            end="\r",
        )

print(f"\n\nGeneration done in {(time.time()-t0)/60:.1f} min")

# ============================================================
# POST-PROCESS + SAVE
# ============================================================
rows, stats = [], {}

for i, text in enumerate(all_generated):
    svg, stage = process_output(text, prompts[i])
    stats[stage] = stats.get(stage, 0) + 1
    rows.append({"id": ids[i], "svg": svg})

pd.DataFrame(rows).to_csv(SUBMISSION_CSV, index=False)

total          = len(rows)
fallback_total = sum(v for k, v in stats.items() if "fallback" in k)

print(f"\n✅ Saved to {SUBMISSION_CSV}")
print(f"\n--- Output Quality Breakdown ---")
for stage, count in sorted(stats.items(), key=lambda x: -x[1]):
    bar = "█" * int(count / total * 40)
    print(f"  {stage:<28} {count:>5}  ({count/total*100:5.1f}%)  {bar}")
print(f"\n  Valid:    {total - fallback_total} ({(total-fallback_total)/total*100:.1f}%)")
print(f"  Fallback: {fallback_total} ({fallback_total/total*100:.1f}%)")