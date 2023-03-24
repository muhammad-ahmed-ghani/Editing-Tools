---
title: Editing Tools
emoji: 📽️📷🎥📹🎦🖼️🎨🖌️
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 3.22.1
app_file: app.py
pinned: false
---

```bash
conda create -n editing-tools python=3.9 -y
conda activate editing-tools
conda install -c "nvidia/label/cuda-11.7.0" cuda-toolkit cuda
pip install -r requirements.txt
python app.py
```

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
