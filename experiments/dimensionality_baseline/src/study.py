"""Training/validation selection and frozen test evaluation for the wafer study."""
from pathlib import Path
import json, time, hashlib, platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix
import joblib

CLASSES=['Center','Donut','Edge-Loc','Edge-Ring','Loc','Near-full','Random','Scratch','none']
CONFIG={'seed':20261021,'pca_dimensions':[1,2,4,8,16,32,64,128],
        'lda_dimensions':list(range(1,9)),'lda_input_dimensions':128,
        'classifiers':['3NN','NCM'],'selection_metric':'validation_macro_f1',
        'bootstrap_repetitions':2000,'pca_whiten':False,'pca_randomized_iter':5}

def metrics(y,p):
    return dict(accuracy=accuracy_score(y,p),balanced_accuracy=balanced_accuracy_score(y,p),
                macro_f1=f1_score(y,p,labels=np.arange(9),average='macro',zero_division=0))

def classifier(name):
    return KNeighborsClassifier(n_neighbors=3,weights='uniform',algorithm='brute',n_jobs=1) if name=='3NN' else NearestCentroid()

def representations(X,pca,lda):
    a=pca.transform(X)
    return {'Raw':X,'PCA':a,'PCA+LDA':lda.transform(a)}

def fit_development(data,root):
    """This function only accesses training and validation arrays."""
    root=Path(root);out=root/'results';out.mkdir(exist_ok=True)
    (out/'config.json').write_text(json.dumps(CONFIG,indent=2))
    x,y=data['X_train'],data['y_train'];v,vy=data['X_validation'],data['y_validation']
    t=time.perf_counter()
    pca=PCA(n_components=128,svd_solver='randomized',iterated_power=5,random_state=CONFIG['seed'],whiten=False)
    z=pca.fit_transform(x)
    lda=LinearDiscriminantAnalysis(solver='svd',n_components=8,priors=np.ones(9)/9)
    l=lda.fit_transform(z,y)
    train={'Raw':x,'PCA':z,'PCA+LDA':l};validation=representations(v,pca,lda)
    rows=[];models={}
    for method in train:
        dims=[4096] if method=='Raw' else CONFIG['pca_dimensions'] if method=='PCA' else CONFIG['lda_dimensions']
        for clf in CONFIG['classifiers']:
            for dim in dims:
                model=classifier(clf).fit(train[method][:,:dim],y)
                p=model.predict(validation[method][:,:dim])
                rows.append(dict(method=method,classifier=clf,dimension=dim,**metrics(vy,p)))
                models[(method,clf,dim)]=model
    scores=pd.DataFrame(rows)
    scores.to_csv(out/'validation_scores.csv',index=False)
    # Resolve exact ties by smaller dimension, then stable method/classifier names.
    order=scores.sort_values(['macro_f1','dimension','method','classifier'],ascending=[False,True,True,True])
    selected=order.iloc[0].to_dict();selected['dimension']=int(selected['dimension'])
    family=scores.sort_values(['macro_f1','dimension'],ascending=[False,True]).groupby(['method','classifier'],sort=False).head(1)
    family.to_csv(out/'validation_selected_by_family.csv',index=False)
    (out/'frozen_selection.json').write_text(json.dumps(selected,indent=2))
    np.save(out/'pca_explained_variance_ratio.npy',pca.explained_variance_ratio_)
    bundle=dict(pca=pca,lda=lda,models=models,selected=selected,train=train,
                fit_seconds=time.perf_counter()-t)
    print(f"Development complete in {bundle['fit_seconds']:.2f} s; validation-selected configuration:",selected)
    return bundle,scores

def fast_macro(y,p):
    cm=np.bincount(y*9+p,minlength=81).reshape(9,9)
    den=cm.sum(0)+cm.sum(1)
    return np.divide(2*np.diag(cm),den,out=np.zeros(9),where=den>0).mean()

def evaluate_frozen(data,bundle,root):
    root=Path(root);out=root/'results';y=data['y_test']
    test=representations(data['X_test'],bundle['pca'],bundle['lda'])
    rows=[];preds={};time_rows=[]
    for (method,clf,dim),model in bundle['models'].items():
        p=model.predict(test[method][:,:dim]);preds[(method,clf,dim)]=p
        rows.append(dict(method=method,classifier=clf,dimension=dim,**metrics(y,p)))
    scores=pd.DataFrame(rows);scores.to_csv(out/'test_curves.csv',index=False)
    chosen=pd.read_csv(out/'validation_selected_by_family.csv')
    final=chosen[['method','classifier','dimension']].merge(scores,on=['method','classifier','dimension'])
    cnn=data['cnn_prediction'];cnn_row=dict(method='Original CNN',classifier='saved predictions',dimension=64,**metrics(y,cnn))
    final=pd.concat([final,pd.DataFrame([cnn_row])],ignore_index=True)
    final.to_csv(out/'common_test_comparison.csv',index=False)
    selected=bundle['selected'];key=(selected['method'],selected['classifier'],int(selected['dimension']))
    p=preds[key]
    pred_frame=pd.DataFrame({'source_index':data['index_test'],'true':y,'selected_baseline':p,'original_cnn':cnn})
    for (method,clf,dim),pr in preds.items():pred_frame[f'{method}_{clf}_{dim}']=pr
    pred_frame.to_csv(out/'test_predictions.csv',index=False)
    for name,pr in [('selected_baseline',p),('original_cnn',cnn)]:
        pd.DataFrame(classification_report(y,pr,labels=np.arange(9),target_names=CLASSES,output_dict=True,zero_division=0)).T.to_csv(out/f'{name}_classification_report.csv')
        np.save(out/f'{name}_confusion.npy',confusion_matrix(y,pr,labels=np.arange(9)))
    # Paired, class-stratified bootstrap on the preselected baseline vs saved CNN.
    rng=np.random.default_rng(CONFIG['seed']);indices=[np.flatnonzero(y==c) for c in range(9)]
    boot=[]
    for _ in range(CONFIG['bootstrap_repetitions']):
        ix=np.concatenate([rng.choice(a,size=len(a),replace=True) for a in indices])
        boot.append(fast_macro(y[ix],p[ix])-fast_macro(y[ix],cnn[ix]))
    diff=fast_macro(y,p)-fast_macro(y,cnn);ci=np.quantile(boot,[.025,.975])
    gain_bal=balanced_accuracy_score(y,p)-balanced_accuracy_score(y,cnn)
    eligible=bool(diff>=.03 and ci[0]>0 and gain_bal>0)
    # Warm end-to-end baseline inference timings, including learned projections.
    q=data['X_test'][:min(100,len(y))]
    def infer(a):
        if key[0]!='Raw':a=bundle['pca'].transform(a)
        if key[0]=='PCA+LDA':a=bundle['lda'].transform(a)
        return bundle['models'][key].predict(a[:,:key[2]])
    infer(q);timings=[]
    for _ in range(5):
        t=time.perf_counter();infer(q);timings.append((time.perf_counter()-t)/len(q)*1000)
    artifact=dict(pca=bundle['pca'] if key[0]!='Raw' else None,
                  lda=bundle['lda'] if key[0]=='PCA+LDA' else None,
                  classifier=bundle['models'][key],selection=selected,classes=CLASSES,
                  preprocessing='64x64 center square-pad, floor-index nearest, divide by 2')
    joblib.dump(artifact,out/'selected_baseline.joblib',compress=3)
    summary=dict(selected_on_validation=selected,selected_test=metrics(y,p),cnn_common_test=metrics(y,cnn),
       macro_f1_delta=float(diff),macro_f1_delta_ci95=ci.tolist(),balanced_accuracy_delta=float(gain_bal),
       integration_eligible=eligible,bootstrap='paired class-stratified percentile bootstrap; sample uncertainty only, not lot-level or training uncertainty',
       inference_median_ms_per_sample=float(np.median(timings)),timing_batch_size=len(q),
       inference_note='Includes PCA/LDA and classifier, excludes image preprocessing and I/O; no comparable CNN timing measured',
       development_seconds=bundle['fit_seconds'],python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,
       cnn_note='Original CNN trained on substantially more samples. Previously observed official test set is reused as a retrospective benchmark, not a new prospective test.',
       comparison_note='Class-capped test subset after exact-input deduplication, not production prevalence; same test indices for every method; no test-driven configuration changes.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    return scores,final,summary
