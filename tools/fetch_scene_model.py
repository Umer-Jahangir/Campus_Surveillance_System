"""Download the pinned public scene checkpoint and verify its SHA-256."""
from pathlib import Path
import hashlib,urllib.request
ROOT=Path(__file__).resolve().parents[1]
url='https://huggingface.co/visionlab-ai/school-violence-detection-models/resolve/a744b6af7496f0cbfa4f0ba32acd46b65e52d4e1/final/final_x3d_realtime.pt'
expected='e833f69d110f167cad4a6c38d385564bdb2f6de63d246e45cb03ff9aa17f0349'
target=ROOT/'src/models/final_x3d_realtime.pt'
if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==expected:
    print('Supported scene weights already present')
else:
    with urllib.request.urlopen(url,timeout=90) as response: data=response.read()
    if hashlib.sha256(data).hexdigest()!=expected: raise ValueError('Downloaded checkpoint hash mismatch; nothing installed')
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(data)
    print('Installed verified-hash scene checkpoint:',target)
