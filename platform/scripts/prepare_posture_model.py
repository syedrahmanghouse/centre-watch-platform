#!/usr/bin/env python3
"""Prepare local pose and CLIP models for posture evidence. This is not training."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import subprocess, sys, tempfile, urllib.request
from datetime import datetime, timezone

PLATFORM=Path(__file__).resolve().parents[1]
MODEL_DIR=Path(os.environ.get('CENTREWATCH_MODELS_DIR',PLATFORM/'models'))
POSE_URL='https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt'
DEFAULTS={'imageSize':960,'poseConfidence':.20,'clipMargin':.012,'clipStandingMargin':.012,'clipMinimumSimilarity':.18,'keypointConfidence':.25}

def download(url,destination):
    if destination.exists(): return
    fd,name=tempfile.mkstemp(prefix='.posture-',suffix='.pt',dir=destination.parent)
    try:
        with os.fdopen(fd,'wb') as target,urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'CentreWatch-Model-Setup/1.0'}),timeout=120) as response:
            size=0
            while chunk:=response.read(1024*1024):
                size+=len(chunk)
                if size>600*1024*1024: raise ValueError('Posture checkpoint exceeds expected size.')
                target.write(chunk)
        Path(name).replace(destination)
    finally: Path(name).unlink(missing_ok=True)

def verify():
    code=r'''
import socket,json,sys,numpy as np
sys.path.insert(0,sys.argv[1]);attempts=[]
def denied(*args,**kwargs):
 attempts.append('network');raise RuntimeError('Posture offline verification blocked network access')
socket.getaddrinfo=denied;socket.create_connection=denied;socket.socket.connect=denied;socket.socket.connect_ex=denied
from server.posture import assign_postures
rows,meta=assign_postures(np.zeros((320,320,3),dtype=np.uint8),[{'label':'person','confidence':.9,'xyxy':[80,30,230,300]}])
assert not attempts
assert rows[0]['posture']=='unknown', 'Blank crop must not establish a posture'
print(json.dumps({'offlineLoadAndInference':True,'networkAttempts':len(attempts)}))
'''
    result=subprocess.run([sys.executable,'-c',code,str(PLATFORM)],check=True,capture_output=True,text=True,timeout=300)
    for line in reversed(result.stdout.splitlines()):
        try:
            value=json.loads(line)
            if value.get('offlineLoadAndInference'): return value
        except (ValueError,AttributeError):pass
    raise RuntimeError('Posture offline verification did not return confirmation.')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    if args.verify_only: print(json.dumps(verify(),indent=2));return
    MODEL_DIR.mkdir(parents=True,exist_ok=True);(MODEL_DIR/'clip').mkdir(exist_ok=True)
    download(POSE_URL,MODEL_DIR/'yolo11m-pose.pt')
    import clip
    # Exact local checkpoint path is used during inference; this explicit setup is the only download path.
    if not (MODEL_DIR/'clip/ViT-B-32.pt').exists():
        clip.load('ViT-B/32',device='cpu',download_root=str(MODEL_DIR/'clip'))
    value={'poseModelFile':'yolo11m-pose.pt','clipModelFile':'clip/ViT-B-32.pt',**DEFAULTS,
           'createdAt':datetime.now(timezone.utc).isoformat(),'poseSource':POSE_URL,
           'clipSource':'https://github.com/openai/CLIP','limitation':'Pretrained posture evidence, not classroom-specific training or role identification.'}
    for key in ('poseModelFile','clipModelFile'):
        value[key.replace('File','Sha256')]=hashlib.sha256((MODEL_DIR/value[key]).read_bytes()).hexdigest()
    sys.path.insert(0,str(PLATFORM));from server.posture import validate_manifest
    validate_manifest(value)
    manifest=MODEL_DIR/'posture.json';temporary=manifest.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(manifest)
    value['offlineVerification']=verify();temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(manifest)
    print(json.dumps({'ready':True,**value['offlineVerification']},indent=2))

if __name__=='__main__': main()
