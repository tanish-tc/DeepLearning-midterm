import pandas as pd
import glob
import os
import json

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
FOLDER_PATH = "././submissions"  # Folder with your 3-4 CSVs
OUTPUT_HTML = "interactive_gallery.html"
TEST_CSV = "./dataset/test.csv" # To pull the prompts

# The file that gets selected by default
PRIORITY_FILE = "visualize-svgs/submission-best.csv" 

# THE FIX: Strip the path so we only compare the actual filename
PRIORITY_FILENAME = os.path.basename(PRIORITY_FILE)

print(f"📂 Loading master prompts from {TEST_CSV}...")
try:
    test_df = pd.read_csv(TEST_CSV)
    gallery_data = {
        row['id']: {"prompt": row['prompt'], "svgs": [], "default_idx": 0} 
        for _, row in test_df.iterrows()
    }
except FileNotFoundError:
    print(f"❌ ERROR: Could not find {TEST_CSV}.")
    exit()

# 1. Grab files from the iterations folder
csv_files = glob.glob(os.path.join(FOLDER_PATH, "*.csv"))

# 2. Add the priority file
if os.path.exists(PRIORITY_FILE):
    print(f"🌟 Found Priority File: {PRIORITY_FILE}")
    # Don't add it twice if it happens to be inside FOLDER_PATH
    if PRIORITY_FILE not in csv_files and os.path.abspath(PRIORITY_FILE) not in [os.path.abspath(f) for f in csv_files]:
        csv_files.append(PRIORITY_FILE)
else:
    print(f"⚠️ WARNING: Could not find {PRIORITY_FILE}!")

print(f"🔍 Loading {len(csv_files)} total submission files...")

# Load SVGs from all files
for file in csv_files:
    filename = os.path.basename(file)
    try:
        df = pd.read_csv(file)
        if 'id' not in df.columns or 'svg' not in df.columns: continue
            
        for _, row in df.iterrows():
            item_id = row['id']
            if item_id in gallery_data:
                gallery_data[item_id]["svgs"].append({
                    "source": filename,
                    "svg": row['svg']
                })
                
                # THE FIX: Compare against the cleaned filename!
                if filename == PRIORITY_FILENAME:
                    gallery_data[item_id]["default_idx"] = len(gallery_data[item_id]["svgs"]) - 1

    except Exception as e:
        print(f"⚠️ Error reading {filename}: {e}")

# Convert to a list and filter out empty ones
js_data = [{"id": k, **v} for k, v in gallery_data.items() if len(v["svgs"]) > 0]

# ==========================================
# 🎨 GENERATE INTERACTIVE HTML
# ==========================================
print("🎨 Building Interactive HTML Interface...")

html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>SVG Master Curation Studio</title>
    <style>
        body {{ font-family: system-ui, sans-serif; background: #eef2f5; padding: 20px; margin: 0; padding-bottom: 100px; }}
        h1 {{ text-align: center; color: #333; }}
        .toolbar {{ position: fixed; bottom: 0; left: 0; right: 0; background: #1a1a1a; padding: 15px; text-align: center; box-shadow: 0 -4px 10px rgba(0,0,0,0.2); z-index: 1000; }}
        .btn-finalize {{ background: #00e676; color: #000; font-weight: bold; padding: 12px 30px; font-size: 18px; border: none; border-radius: 8px; cursor: pointer; transition: transform 0.2s; }}
        .btn-finalize:hover {{ transform: scale(1.05); }}
        
        .prompt-section {{ background: white; margin-bottom: 30px; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
        .prompt-text {{ font-size: 16px; color: #222; margin-bottom: 15px; font-weight: bold; }}
        
        .gallery {{ display: flex; flex-wrap: wrap; gap: 20px; }}
        .card {{ border: 3px solid #eee; border-radius: 10px; cursor: pointer; padding: 10px; transition: all 0.2s; background: #fafafa; position: relative; }}
        .card:hover {{ border-color: #bbb; }}
        
        .card.selected {{ border-color: #00e676; background: #e8fbed; box-shadow: 0 4px 12px rgba(0,230,118,0.3); }}
        .card.selected::after {{ content: '✅ SELECTED'; position: absolute; top: -12px; right: -12px; background: #00e676; color: black; font-size: 12px; font-weight: bold; padding: 4px 8px; border-radius: 12px; }}
        
        .svg-container {{ width: 256px; height: 256px; background: #fff; display: flex; align-items: center; justify-content: center; overflow: hidden; border-radius: 4px; pointer-events: none; }}
        .source-tag {{ font-size: 12px; color: #666; margin-top: 10px; text-align: center; pointer-events: none; font-weight: 500; }}
        
        .priority-tag {{ color: #00a152; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Kaggle SVG Curation Studio</h1>
    <div id="app"></div>

    <div class="toolbar">
        <button class="btn-finalize" onclick="downloadFinalCSV()">💾 Finalize & Download Submission</button>
    </div>

    <script>
        const galleryData = {json.dumps(js_data)};
        const selections = {{}}; 
        const PRIORITY_FILENAME = "{PRIORITY_FILENAME}";

        function renderGallery() {{
            const app = document.getElementById('app');
            let html = '';

            galleryData.forEach((item, itemIndex) => {{
                selections[item.id] = item.default_idx;

                html += `<div class="prompt-section" id="section-${{item.id}}">
                            <div class="prompt-text">ID: ${{item.id}} | <strong>Prompt:</strong> ${{item.prompt}}</div>
                            <div class="gallery">`;

                item.svgs.forEach((svgData, idx) => {{
                    const isSelected = (idx === item.default_idx) ? 'selected' : '';
                    const isPriority = (svgData.source === PRIORITY_FILENAME) ? 'priority-tag' : '';
                    const tagPrefix = (svgData.source === PRIORITY_FILENAME) ? '🌟 ' : '';
                    
                    html += `
                        <div class="card ${{isSelected}}" id="card-${{item.id}}-${{idx}}" onclick="selectSvg('${{item.id}}', ${{idx}})">
                            <div class="svg-container">${{svgData.svg}}</div>
                            <div class="source-tag ${{isPriority}}">${{tagPrefix}}${{svgData.source}}</div>
                        </div>`;
                }});

                html += `   </div>
                         </div>`;
            }});
            app.innerHTML = html;
        }}

        function selectSvg(itemId, selectedIdx) {{
            selections[itemId] = selectedIdx;
            const section = document.getElementById(`section-${{itemId}}`);
            const cards = section.querySelectorAll('.card');
            cards.forEach(card => card.classList.remove('selected'));
            document.getElementById(`card-${{itemId}}-${{selectedIdx}}`).classList.add('selected');
        }}

        function downloadFinalCSV() {{
            let csvContent = "id,svg\\n";
            
            galleryData.forEach(item => {{
                const selectedIdx = selections[item.id];
                const finalSvg = item.svgs[selectedIdx].svg;
                const escapedSvg = finalSvg.replace(/"/g, '""');
                csvContent += `${{item.id}},"${{escapedSvg}}"\\n`;
            }});

            const blob = new Blob([csvContent], {{ type: 'text/csv;charset=utf-8;' }});
            const link = document.createElement("a");
            const url = URL.createObjectURL(blob);
            link.setAttribute("href", url);
            link.setAttribute("download", "human_curated_submission.csv");
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }}

        renderGallery();
    </script>
</body>
</html>'''

with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"✅ Interactive Visualizer generated at {OUTPUT_HTML}")