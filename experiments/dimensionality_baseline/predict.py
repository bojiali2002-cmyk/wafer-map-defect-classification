"""Predict a saved .npy raw wafer map with the validation-selected baseline.
python predict.py path/to/raw_wafer.npy
Only load joblib models and NumPy files from trusted sources.
"""
from pathlib import Path
import argparse,joblib,numpy as np
from src.prepare_data import preprocess

def predict_maps(maps,model_path=None):
    root=Path(__file__).resolve().parent
    model=joblib.load(model_path or root/'results/selected_baseline.joblib')
    x=np.stack([preprocess(m).ravel() for m in maps])
    if model['pca'] is not None:x=model['pca'].transform(x)
    if model['lda'] is not None:x=model['lda'].transform(x)
    p=model['classifier'].predict(x[:,:int(model['selection']['dimension'])])
    return [model['classes'][i] for i in p]

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('wafer_npy');args=parser.parse_args()
    print(predict_maps([np.load(args.wafer_npy,allow_pickle=False)])[0])
