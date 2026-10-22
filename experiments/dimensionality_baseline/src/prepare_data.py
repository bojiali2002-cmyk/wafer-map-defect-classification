"""Create the portable study inputs from original WM-811K and project artifacts.
Run once with --raw, --original, --output. Normal notebook use needs no raw data.
"""
from pathlib import Path
import argparse, hashlib, json, pickle, shutil
import numpy as np
import pandas as pd

CLASSES = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Near-full','Random','Scratch','none']
class LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(module.replace('pandas.indexes','pandas.core.indexes'), name)

def preprocess(wafer, size=64):
    """Match original torch nearest interpolation (floor index), padding and /2.
    PIL nearest uses a different coordinate convention and is not substituted.
    """
    a = np.asarray(wafer, dtype=np.float32)
    h,w = a.shape; side=max(h,w)
    square=np.zeros((side,side), dtype=np.float32)
    square[(side-h)//2:(side-h)//2+h, (side-w)//2:(side-w)//2+w]=a
    index=np.floor(np.arange(size)*side/size).astype(int)
    return square[index[:,None],index[None,:]]/2

def prepare(raw, original, output):
    original, output=Path(original),Path(output);output.mkdir(parents=True,exist_ok=True)
    with open(raw,'rb') as f: df=LegacyUnpickler(f,encoding='latin1').load()
    predictions=pd.read_csv(original/'models/test_predictions.csv').set_index('source_index')
    seen=set(); arrays={}; manifests=[]; audit={}
    for split,cap in [('train',500),('validation',100),('test',200)]:
        full=pd.read_csv(original/f'data/processed/{split}_manifest.csv')
        selected=pd.concat([g.sample(min(cap,len(g)),random_state=20261021) for _,g in full.groupby('label_index')])
        selected=selected.sort_values('source_index').reset_index(drop=True)
        keep=[];x=[];removed=0
        for i,r in selected.iterrows():
            source=int(r.source_index)
            # Check source alignment with original label and lot metadata.
            label=np.asarray(df.at[source,'failureType']).ravel()[0]
            assert label==r.failure_type and df.at[source,'lotName']==r.lotName
            image=preprocess(df.at[source,'waferMap'])
            digest=hashlib.sha256(image.tobytes()).hexdigest()
            if digest in seen: removed+=1;continue
            seen.add(digest);keep.append(i);x.append(image)
        selected=selected.iloc[keep].copy();selected['image_sha256']=[hashlib.sha256(a.tobytes()).hexdigest() for a in x]
        arrays[f'X_{split}']=np.stack(x).reshape(len(x),-1)
        arrays[f'y_{split}']=selected.label_index.to_numpy(dtype=np.int64)
        arrays[f'index_{split}']=selected.source_index.to_numpy(dtype=np.int64)
        if split=='test':
            p=predictions.loc[selected.source_index]
            assert np.array_equal(p.label_index.to_numpy(),arrays['y_test'])
            arrays['cnn_prediction']=p.predicted_index.to_numpy(dtype=np.int64)
        manifests.append(selected)
        audit[split]={'before_dedup':len(keep)+removed,'after_dedup':len(keep),'removed_duplicate_images':removed,'classes':selected.failure_type.value_counts().to_dict(),'lots':int(selected.lotName.nunique())}
    for i,a in enumerate(manifests):
        for b in manifests[i+1:]:assert not set(a.lotName)&set(b.lotName)
    manifest=pd.concat(manifests,ignore_index=True)
    manifest.to_csv(output/'manifest.csv',index=False)
    np.savez_compressed(output/'study_data.npz',**arrays)
    shutil.copyfile(original/'models/final_metrics.json',output/'original_cnn_full_metrics.json')
    audit.update(seed=20261021,classes=CLASSES,preprocessing='center zero-pad to square; floor-index nearest resize 64x64; divide by 2',raw_sha256=hashlib.file_digest(open(raw,'rb'),'sha256').hexdigest(),subset_sha256=hashlib.file_digest(open(output/'study_data.npz','rb'),'sha256').hexdigest(),source='https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map',split_policy='preserve original project train/validation/test manifests; independent within-class caps; remove exact duplicates of final inputs across and within selected splits, train then validation then test priority',cnn_provenance='previously saved original CNN predictions, not a newly trained CNN; trained with more data; same test source indices and labels checked',integration_rule='Only consider adding a baseline if validation-selected method has >=0.03 macro-F1 gain over CNN, positive lower 95% paired bootstrap bound, and higher balanced accuracy on the common test subset. No CNN replacement or deployment claim.')
    (output/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--original',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();prepare(a.raw,a.original,a.output)
