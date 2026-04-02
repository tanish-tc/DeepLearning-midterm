# 🎨 Deep Learning - Kaggle Mid-Term: The SVG Generation Chronicles

**Final Kaggle Score:** `16.05142` (and climbing)  
**Hardware Hopping:** Colab (Single A100) ➡️ Chameleon Cloud (4x A100 Cluster) ➡️ Apple Silicon (MLX)  
**Core Stack:** PyTorch, Hugging Face `transformers`, `vLLM`, `mlx-lm`, `xml.etree.ElementTree`

## 📖 The Prologue: Deceptively Simple
The objective of this competition sounded easy enough: fine-tune a Large Language Model to act as a vector graphics engine. Feed it a text prompt, and get perfectly structured, mathematically sound SVG code in return.

But Kaggle sets traps. 
1. **The 8,000 Character Hard Limit:** Go over this, and your row gets a `0.0`. 
2. **The `256x256` Canvas:** The output must strictly adhere to `viewBox="0 0 256 256"`.
3. **The XML Parser of Doom:** If the model hallucinates a single non-existent tag, misses a closing `</path>`, or drops a quote, the strict XML parser crashes the entire evaluation.

This repository is the log of how we survived dependency hell, bypassed VRAM bottlenecks, hacked inference engines, and engineered a human-in-the-loop multi-model ensemble to climb the leaderboard.

---

## 🔬 The Iteration Log: From Bugs to Breakthroughs

### Iteration 1: Humble Beginnings (The 1.5B 4-bit Era)
* **Artifacts:** `models/Qwen2.5-1.5B-4bit-merged.zip`
* **Config:** `r=32`, `alpha=64`, `LR=1e-4`, 4-bit Quantized

We started lean. To fit the training loop into a standard Colab instance, we took Qwen2.5-1.5B and quantized it to 4-bit. We fired off two massive overnight runs just to get the loss curve to stabilize. 

**The Reality Check:** The model learned the shapes, but it was incredibly sloppy. It hallucinated non-existent attributes like `filling="0"` and babbled out massive coordinates like `M20.833396911621094 100.0`. Out of 1000 prompts, 560 completely failed the 8k character limit or XML parsing. However, the 440 that *did* pass were mathematically perfect, carrying our initial baseline score to **15.30**.

> 📸 **[INSERT IMAGE HERE: Screenshot of the first Colab notebook showing the 560/1000 fallback warning]**

### Iteration 2: The "Data Wash" & Scaling Up
* **Artifacts:** `models/Qwen2.5-1.5B-merged.zip` ➡️ `models/Qwen2.5-1.5B-30+trains.zip`

Quantization was degrading the precise syntax required for XML. We moved to pure BFloat16 and ran over 30 training iterations (`30+ trains.zip`), tweaking the learning rate and warmup steps. 

It was here we implemented **"The Data Wash"** (`test_submission.py`). We realized we were wasting thousands of tokens on decimal precision. We wrote a regex pipeline to pre-process the dataset, truncating all SVG coordinates to 2 decimal places (`100.12`). We also hard-coded fixes to scrub empty `fill=""` attributes and enforce the 256x256 header. This functionally tripled the amount of geometric paths we could fit into the 8k limit.

> 📸 **[INSERT IMAGE HERE: Side-by-side comparison of the raw 15-decimal dataset vs the washed 2-decimal dataset]**

### Iteration 3: Hardware Hopping (Chameleon 4x A100 & MLX)
* **Artifacts:** `models/gemma3-weights.zip`

Colab was getting too slow for our ambitions. We migrated the pipeline to a Chameleon Cloud cluster wielding 4x A100s. 

We tried fine-tuning **Gemma 3 (4B)**. It was incredibly smart—it respected the `256x256` viewBox perfectly. But there was a massive bug: `vLLM`'s internal engine panicked because Gemma 3 is a multimodal architecture, and it crashed looking for vision encoders. 

**The Pivot:** We abandoned vLLM for Gemma and wrote a highly optimized, dynamically batched PyTorch inference script (`batch_size=32`, left-padding). Simultaneously, to keep testing locally, we built a robust MLX inference pipeline (`local-inference/inference.py`) to run sanity checks and degenerate-path pruning directly on Apple Silicon.

### Iteration 4: The 3B Heavyweights & vLLM Hell
* **Artifacts:** `models/Qwen3.5-3B-weights.zip`, `models/Qwen2.5-3B_4bit-weights.zip`, `models/Qwen2.5-3B-weights.zip`
* **Score:** **16.05142**

We scaled up the base model to Qwen 3B in pure BFloat16 with DDP (Distributed Data Parallel) across the 4x A100s. The model possessed a deep understanding of spatial geometry, securing our breakthrough **16.05** score.

But inference was a nightmare. We wanted to run a **Best-of-5 (`n=5`)** sampling strategy to catch failures. When we booted up vLLM on Colab, injecting LoRA weights while running `n=5` caused massive VRAM fragmentation. `EngineDeadError` became our sleep paralysis demon. 

**The "Flat Batch" Hack:** Instead of using vLLM's buggy internal `n=5` parameter, we hacked the input list. We wrote a script that literally duplicated every prompt 5 times in the array and ran vLLM with `n=1`. It bypassed the memory bug completely and generated 5,000 outputs in minutes. 

> 📸 **[INSERT IMAGE HERE: Kaggle Leaderboard screenshot showing the 16.05142 score]**

---

## 🧠 The Secret Weapons: Ensembling & Ablations

### The Priority Mega-Tournament
We had a folder (`submission-iterations/`) filled with 15+ CSVs from different models, epochs, and temperatures. Why rely on just one?

We wrote `priority_mega_ensemble.py`. 
1. It loads all historical CSVs.
2. It parses the SVG for every single prompt.
3. It literally counts the physical XML tags (`len(list(root.iter()))`) to find the most geometrically complex drawing.
4. **The Tie-Breaker:** We gave our best Qwen-3B outputs "Priority Status." If an older model tied with our new model, the script automatically swapped in the newer, mathematically cleaner code based on shortest string length.

### The "God-Mode" Visualizer
Algorithms are great, but the human eye is the ultimate judge. We built `inter-visualize.py`—a Single Page Application (SPA) in Python/HTML. It loads the merged ensemble data and displays a beautiful interactive gallery. We could scroll through, see the model outputs side-by-side, manually click the best one (highlighting it in green), and click a button to instantly download the final, human-curated `submission.csv`.

> 📸 **[INSERT IMAGE HERE: The Interactive HTML "God-Mode" UI showing the ✅ SELECTED SVGs]**

### Ablation Studies
To mathematically prove *why* our parameters worked, we ran an ablation study on the first 5 prompts using 4 different configurations.
* **Baseline (Greedy):** `Temp 0.0` (Too simplistic).
* **High Temp:** `Temp 0.8` (Disaster. Hallucinated broken XML tags).
* **Strict Penalty:** `Temp 0.35, Repetition Penalty 1.2` (Stopped the infinite drawing loops).
* **The Champion:** `Temp 0.35, Rep 1.1, Best-of-5` (The sweet spot of creativity and structural integrity).

> 📸 **[INSERT IMAGE HERE: Screenshot of the ablation_matrix.html table showing the failed high-temp runs vs the successful Champion runs]**

---

## 📁 Repository Structure (Detailed)

### Root Files
* `.gitignore`
  * Ignores large model archive files under `models/` and ignores the full `dataset/` directory by default to keep the git history clean.
* `notebook-test1.ipynb`
  * A large starter/experimentation notebook for LoRA fine-tuning and inference prep. Uses Unsloth + Hugging Face style workflow.
* `notebook-test2.ipynb`
  * A cleaner end-to-end notebook for training, inference, and pruning. Includes data quality gate logic, multi-source dataset loading, and model switch configuration (`1.5B`, `7B`, `deepseek-6.7B`).
* `README.md`
  * You are here.

### `dataset/`
* `dataset/train.csv` - Raw training set (`id,prompt,svg`).
* `dataset/test.csv` - Test prompts (`id,prompt`) containing 1000 evaluation rows.
* `dataset/together_train_washed.jsonl` - Cleaned, chat-formatted JSON objects produced by our data washing pipeline.

### `local-inference/`
* `local-inference/inference.py`
  * The most robust local **MLX** inference pipeline in this repo. Applies multi-stage post-processing (loop truncation, degenerate duplicate pruning, regex extraction, XML repair) and injects fallback SVGs for unrecoverable outputs.
* `local-inference/run_vllm_inference.py`
  * Simpler MLX script (despite the filename). Uses basic extract/validate/fallback flow.
* `local-inference/run_vllm_inference_v2.py`
  * True **vLLM** batch inference script designed for A100-class hardware. Uses tuned sampling parameters and stop tokens to reduce looping.

### `models/`
*(Note: Model weight archives are all `.zip` and intentionally gitignored. This is where we store our checkpoints before deploying to Kaggle).*
* `Qwen2.5-1.5B-4bit-merged.zip` | `Qwen2.5-1.5B-merged.zip` | `Qwen2.5-1.5B-30+trains.zip`
* `gemma3-weights.zip`
* `Qwen3.5-3B-weights.zip` | `Qwen2.5-3B_4bit-weights.zip` | `Qwen2.5-3B-weights.zip`

### `submission-iterations/`
*The experiment log of output CSVs and evolved ensemble candidates.*
* `test_submission.py` - The Data Wash script. Truncates floats, fixes `fill=""`, removes `filling`, and drops `>7000` char SVGs.
* `submission-v1.csv` ... `submission-v15.csv` - Iterative generations.
* `submission-gemma.csv` - Gemma-based run output.
* `submission-best-washed.csv` / `submission-best-mathematically_scaled.csv` - Variants emphasizing compactness/scaling tweaks.
* `mega_tournament_submission.csv` - Tournament-style final candidate.
* `priority_mega_ensemble.csv` - The priority-weighted ensemble output.
* `smart_ensemble_submission_v2.csv` - Updated ensemble strategy output.
* `submission-prompts.csv` - Includes prompts for analysis/visualization context.

### `visualize-svgs/`
* `visualize.py` - Reads a submission CSV and generates a static HTML gallery of SVG cards.
* `inter-visualize.py` - Builds the interactive multi-submission curation SPA in HTML. Supports selecting one SVG per prompt and downloading a curated CSV directly from the browser.
* **Generated Artifacts:** `interactive_gallery_compare.html`, `svg_gallery_ensemble.html`, `svg_gallery_gemma.html`, etc.

---

## 🚀 Quick Start (Minimal)

**1. Install Core Dependencies:**
```bash
pip install pandas lxml mlx-lm vllm transformers peft
```

**2. Run robust MLX inference (Apple Silicon):**
```bash
python local-inference/inference.py
```

**3. Generate the Interactive Curation Gallery:**
```bash
python visualize-svgs/inter-visualize.py
```
*(Open the resulting HTML file in Chrome/Safari to curate your ensemble).*

---

### Midterm Summary
This codebase reflects a highly iterative competition strategy:
1. **Data Quality First:** Scrub the data to reduce invalid SVG behavior.
2. **Standardization:** Enforce strict competition constraints (256x256, 8k chars).
3. **Multi-Backbone Testing:** Qwen vs. Gemma vs. DeepSeek.
4. **Bulletproof Inference:** Post-processing recovery via `xml.etree.ElementTree`.
5. **The Final Polish:** Ensemble + human-in-the-loop curation to squeeze out every possible decimal point of accuracy.

