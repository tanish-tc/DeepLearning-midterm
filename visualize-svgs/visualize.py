import pandas as pd

print("📂 Loading submission file...")
df = pd.read_csv('/Users/tanish-tc/Documents/dl-midterm/visualize-svgs/submission-prompts.csv')

# The HTML and CSS framework
html_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>SVG Evaluation Gallery</title>
    <style>
        body { font-family: system-ui, sans-serif; background: #eef2f5; padding: 20px; }
        h1 { text-align: center; color: #333; }
        .gallery { display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }
        .card { background: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); width: 256px; transition: transform 0.2s; }
        .card:hover { transform: scale(1.02); }
        .svg-container { width: 256px; height: 256px; border: 1px solid #ddd; background: #fff; display: flex; align-items: center; justify-content: center; overflow: hidden; }
        .prompt { font-size: 13px; color: #444; margin-top: 15px; max-height: 80px; overflow-y: auto; line-height: 1.4; }
    </style>
</head>
<body>
    <h1>Kaggle SVG Generation Gallery (16.05 Score)</h1>
    <div class="gallery">
'''

print("🎨 Building HTML Grid...")
# Show the first 100 SVGs (showing all 1000 might make the browser lag)
for idx, row in df.head(100).iterrows(): 
    prompt = str(row['prompt']).replace('"', '&quot;') # Escape quotes for HTML
    svg = str(row['svg'])
    
    html_content += f'''
        <div class="card">
            <div class="svg-container">{svg}</div>
            <div class="prompt" title="{prompt}"><strong>Prompt:</strong> {prompt}</div>
        </div>
    '''

html_content += '''
    </div>
</body>
</html>
'''

# Save the file
output_file = 'svg_gallery.html'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"✅ Gallery saved as {output_file}!")
print("Download this file to your computer and open it in Chrome/Safari to view the SVGs.")