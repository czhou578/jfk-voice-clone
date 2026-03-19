import json
import os

req_path = "requirements.txt"
with open(req_path, "r", encoding="utf-8") as f:
    reqs = f.readlines()

# Filter out local conda build paths which will crash Colab's pip
clean_reqs = [r for r in reqs if "@ file://" not in r and r.strip()]

writefile_src = ["%%writefile requirements.txt\n"] + clean_reqs

with open("prepare_tts_colab.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

new_cells = []
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    
    if "!pip install yt-dlp" in src:
        # Insert the requirements.txt generation cell right before the install cell
        new_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": writefile_src
        })
        
        # Update the pip install line
        new_source = []
        for line in c["source"]:
            if line.startswith("!pip install yt-dlp"):
                new_source.append("!pip install -r requirements.txt\n")
            else:
                new_source.append(line)
        c["source"] = new_source
        new_cells.append(c)
    else:
        new_cells.append(c)

nb["cells"] = new_cells

with open("prepare_tts_colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
