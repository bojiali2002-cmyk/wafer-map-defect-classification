"""Execute the complete notebook with the current Python, using a temporary kernel.
Run from any directory: python /path/to/wafer_dimensionality_study/run_notebook.py
"""
from pathlib import Path
import json, os, sys, tempfile
import nbformat
from nbclient import NotebookClient

def main():
    root=Path(__file__).resolve().parent
    path=root/'notebooks/01_complete_study.ipynb'
    nb=nbformat.read(path,as_version=4)
    with tempfile.TemporaryDirectory(prefix='wafer-kernel-') as temp:
        spec=Path(temp)/'kernels'/'wafer-local';spec.mkdir(parents=True)
        (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Wafer local Python','language':'python'}))
        old=os.environ.get('JUPYTER_PATH')
        os.environ['JUPYTER_PATH']=temp+(os.pathsep+old if old else '')
        try:
            NotebookClient(nb,timeout=1200,kernel_name='wafer-local',resources={'metadata':{'path':str(root)}}).execute()
        finally:
            nbformat.write(nb,path)
            if old is None:os.environ.pop('JUPYTER_PATH',None)
            else:os.environ['JUPYTER_PATH']=old
    print('Notebook completed:',path)
if __name__=='__main__':main()
