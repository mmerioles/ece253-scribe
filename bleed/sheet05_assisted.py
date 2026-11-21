"""Sheet05 assisted-registration experiment; leaves automatic batch unchanged.
Corners were manually selected on upright 600x800 previews (TL,TR,BR,BL).
Reverse coordinates refer to the horizontally mirrored back. Filtering defaults
are unchanged; NCC alignment search is widened to 96 native pixels.
"""
import sys,json,contextlib,hashlib
from pathlib import Path
from unittest.mock import patch
import cv2,numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'bleed')]
import recover_overlap as ro
SOURCE=np.float32([[30,24],[582,16],[592,771],[0,767]])
TARGETS=[[[20,47],[557,40],[600,797],[2,800]],[[28,29],[549,37],[590,736],[24,778]],[[36,61],[559,53],[594,784],[12,784]]]
VARIANTS=['input','rowley_brooke','minimal_protection_off','minimal']
OUT=ROOT/'bleed/outputs/sheet05-assisted'

def process(i):
 target=OUT/f'front-{i:02}'; target.mkdir(parents=True,exist_ok=True)
 def register(front,back,already_mirrored=False):
  if already_mirrored: raise ValueError('This experiment expects original back')
  back=back[:,::-1].copy(); h,w=front.shape[:2]
  src=SOURCE*np.array([back.shape[1]/600,back.shape[0]/800],np.float32)
  dst=np.float32(TARGETS[i-1])*np.array([w/600,h/800],np.float32)
  H=cv2.getPerspectiveTransform(src,dst)
  aligned=cv2.warpPerspective(back,H,(w,h),borderValue=(255,255,255))
  valid=cv2.warpPerspective(np.ones(back.shape[:2],np.uint8),H,(w,h),flags=cv2.INTER_NEAREST)>0
  return aligned,valid,dict(method='manual paper corners',homography=H.tolist(),source_points=src.tolist(),target_points=dst.tolist(),note='Approximate corners, including extrapolated cropped bottom edge on front01; not automatic SIFT.')
 original_refine=ro.refine_registration
 def refine(f,b,v): return original_refine(f,b,v,radius=96)
 cv2.setRNGSeed(0)
 with (target/'processing.log').open('w') as log,contextlib.redirect_stdout(log),patch.object(ro,'register',register),patch.object(ro,'refine_registration',refine):
  report,state,result=ro.run(ROOT/f'bleed/inputs/sheet05/front-{i:02}.jpg',ROOT/'bleed/inputs/sheet05/back.jpg',target,save='minimal')
 del state,result
 report['alignment_search_radius']=96
 (target/'recover_run.json').write_text(json.dumps(report,indent=2))
 preview=Image.new('RGB',(1800,830),'white'); draw=ImageDraw.Draw(preview)
 for j,(v,label) in enumerate(zip(['input','rowley_brooke','minimal'],['Raw','Rowley-Brooke (shared alignment)','Improved'])):
  with Image.open(target/f'{v}.png') as im: preview.paste(im.resize((600,800)),(j*600,30))
  draw.text((j*600+10,8),label,fill='black')
 preview.save(target/'comparison.jpg',quality=92)
 with Image.open(target/'input.png') as f,Image.open(target/'aligned_back_dense.png') as b:
  f=np.array(f.resize((600,800)).convert('L')); b=np.array(b.resize((600,800)).convert('L'))
  Image.fromarray(np.stack([b,f,f],-1)).save(target/'alignment.png')
 print(i,report['dense_registration'],flush=True)

def score():
 for i in range(1,4):
  for v in VARIANTS:
   if not (OUT/f"front-{i:02}"/f"{v}.png").exists():
    raise FileNotFoundError("Run restoration before --score")
 import torch
 from transformers import TrOCRProcessor,VisionEncoderDecoderModel
 from jiwer import cer,wer
 from main import MODEL,REVISION,evaluate,read_inputs,normalize
 torch.set_num_threads(4)
 processor=TrOCRProcessor.from_pretrained(MODEL,revision=REVISION,use_fast=False,local_files_only=True)
 model=VisionEncoderDecoderModel.from_pretrained(MODEL,revision=REVISION,use_safetensors=True,local_files_only=True).eval()
 totals={v:([],[]) for v in VARIANTS}; report={'assisted':True,'model':MODEL,'revision':REVISION,'captures':{},'scores':{},'source_sha256':{name:hashlib.sha256((ROOT/'bleed'/name).read_bytes()).hexdigest() for name in ['sheet05_assisted.py','remove_bleed.py','recover_overlap.py']},'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'bleed/inputs/sheet05').glob('*') if p.is_file()}}
 for i in range(1,4):
  target=OUT/f'front-{i:02}'; refs,boxes=read_inputs(ROOT/f'bleed/inputs/sheet05/front-{i:02}.jpg'); scores={}
  for v in VARIANTS:
   with (target/f'ocr-{v}.log').open('w') as log,contextlib.redirect_stdout(log):
    _,pred=evaluate(target/f'{v}.png',refs,boxes,processor,model,target/'ocr'/v)
   a,b=list(map(normalize,refs)),list(map(normalize,pred)); totals[v][0].extend(a); totals[v][1].extend(b)
   scores[v]={'cer':cer(a,b),'wer':wer(a,b)}
  report['captures'][target.name]=scores; print(target.name,scores,flush=True)
 for v,(a,b) in totals.items(): report['scores'][v]={'cer':cer(a,b),'wer':wer(a,b),'lines':len(a)}
 (OUT/'summary.json').write_text(json.dumps(report,indent=2)); print(report['scores'],flush=True)
if __name__=='__main__':
 if '--score' in sys.argv: score()
 else:
  for i in range(1,4): process(i)
