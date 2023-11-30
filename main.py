'''Create musl-1.2.4 dataset, train models, report accuracy.'''

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from multiprocessing import cpu_count
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from tqdm import tqdm
from typing import Dict,List,Tuple
import _utils as U
import argparse,shutil,sys
import gzk.model as zk
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
import skl.forest as rf
import tig.gen as tg
import time
import tok.capdis as cd
import tok.predict as ft


random.seed(42) #reproducible results!
# The reason for this magic constant is explained in tig/README.org
# If we used a different ISA, this number might need to change.
# TODO: determine this number based on whether "nm" shows the binary contains the function.
MIN_OBJ_SIZE = 1675 # still a magic number, but now it has a *name*
ft.fasttext.FastText.eprint = lambda x: None  #silence fastText warnings


def strip(objdir:Path, stripdir:Path) -> (int,int):
  '''discard files that do not contain their namesake'''
  nkept = 0
  nobs = list(objdir.glob('*.o'))
  objs = filter(lambda x:x.stat().st_size>MIN_OBJ_SIZE, nobs)
  for i,f in tqdm(enumerate(objs)):
    out = U.sh(f'nm -af just-symbols {f}').stdout
    if U.fname(f) in out.replace(f.with_suffix('.c').name,''): #function present after removing filename?
      nkept += 1
      U.sh(f'strip {f} -o {stripdir/f.name}')

  return len(nobs),i,nkept


def run_kn(train:List[Path],test:List[Path],name:str):
  '''train deflate+kNN model and predict on test data'''
  t0,t1 = time.time(),0
  knn_ranks = zk.do_nn_future(train,test) #"train"
  t1 = time.time() - t0  #cold start train time
  ytr,yte = ([U.fname(x) for x in t] for t in (train,test))
  results,f1,preds,params = [],[],[],(1,3,5,9)
  for k in params:
    predicted = [Counter(x).most_common(1)[0][0] for x in np.array(ytr)[knn_ranks[:,:k]]] # test
    preds.append(predicted)
    f1score = f1_score(yte,predicted,zero_division=0,labels=yte,average='macro') # measure
    f1.append(f1score)
    results.append([name,'kn',len(train),len(test),0,t1,f1score,f'neighbors={k}'])

  # best f1 score
  i = f1.index(max(f1))
  U.heatmap(f'../paper/images/{name}-kn-{len(train)}-{len(test)}-{params[i]}.pdf', *U.confuse(yte,preds[i]))
  return results


def run_rf(train:List[Path],test:List[Path],name:str):
  '''train Random Forest classifier and predict on test data'''
  data = U.bag_of_bytes(train+test)
  a,b = len(train),len(test)
  Xtr,Xte = data[:a],data[a:]
  ytr,yte = ([U.fname(x) for x in d] for d in (train,test))
  results,f1,preds,params = [],[],[],(40,80,100,120)
  for k in params:
    model = RandomForestClassifier(n_estimators=k,class_weight='balanced',n_jobs=-1)
    t0 = time.time(); model.fit(Xtr,ytr); t1 = time.time() - t0 # train
    t2 = time.time(); predicted = model.predict(Xte); t3 = time.time() - t2 # test
    f1score = f1_score(yte,predicted,zero_division=0,labels=yte,average='macro') # measure
    f1.append(f1score)
    preds.append(predicted)
    results.append([name,'rf',len(train),len(test),t1,t3,f1score,f'estimators={k}'])

  i = f1.index(max(f1))
  U.heatmap(f'../paper/images/{name}-rf-{len(train)}-{len(test)}-{params[i]}.pdf', *U.confuse(yte,preds[i].tolist()))
  return results


def run_rt(train:List[Path],test:List[Path],name:str):
  '''train Random Forest on disassembly tokens rather than bytes'''
  # TODO: Why worse than raw bytes? Maybe lose context like "what kind of mov is this?"
  # TODO: imshow both mattrain and mattest side-by-side
  a = len(train)
  data = cd.disassemble(train+test)
  Xtr,Xte = data[:a],data[a:]
  ytr,yte = ([U.fname(x) for x in t] for t in (train,test))
  vocab = U.make_vocab(' '.join(Xtr))
  sz = max(vocab.values())+1
  mattrain = np.zeros((len(Xtr),sz),np.int32)
  mattest = np.zeros((len(Xte),sz),np.int32)
  for i,X in enumerate(Xtr):
    for j in U.tti(X,vocab):
      mattrain[i,j] += 1

  for i,X in enumerate(Xte):
    for j in U.tti(X,vocab):
      mattest[i,j] += 1

  # return mattrain,mattest
  trunc = min(mattrain.shape[1],mattest.shape[1]) #in some cases (like train:obfuscated/test:plain) these are equal.
  results,f1,preds,params = [],[],[],(40,80,100,120)
  for k in params:
    model = RandomForestClassifier(n_estimators=k,class_weight='balanced',n_jobs=-1)
    t0 = time.time(); model.fit(mattrain[:,:trunc],ytr); t1 = time.time() - t0 # train
    t2 = time.time(); predicted = model.predict(mattest[:,:trunc]); t3 = time.time() - t2 # test
    f1score = f1_score(yte,predicted,zero_division=0,labels=yte,average='macro') # measure
    f1.append(f1score)
    preds.append(predicted)
    results.append([name,'rt',len(train),len(test),t1,t3,f1score,f'estimators={k}'])

  i = f1.index(max(f1))
  U.heatmap(f'../paper/images/{name}-rt-{len(train)}-{len(test)}-{params[i]}.pdf', *U.confuse(yte,preds[i].tolist()))
  return results


def run_ft(train:List[Path],test:List[Path],name:str):
  '''train FastText model and predict on test data'''
  ytr,yte = ([U.fname(x) for x in t] for t in (train,test))
  # tokenize train+test data in one big file - use original sizes to split later
  a = len(train)
  data = cd.disassemble(train+test)
  Xtr,Xte = data[:a],data[a:]
  tt = time.time()
  fttrain = Path('/tmp')/f'ft-{a}.train'
  with open(fttrain,'w') as f:
    for line in Xtr:
      print(line,file=f)

  model = ft.fasttext.train_supervised(input=str(fttrain),thread=18,lr=2,epoch=300,dim=30,wordNgrams=5,verbose=0)
  tt = time.time() - tt
  anon = [x[x.index(' '):] for x in Xte]  #remove labels
  results,f1,preds,params = [],[],[],(0.0,0.1,0.3,0.7)
  for t in params:
    t0 = time.time()
    predicted = [x[0][9:] if x else '[None]' for x in model.predict(anon, k=1, threshold=t)[0]]
    t1 = time.time() - t0
    f1score = f1_score(yte,predicted,zero_division=0,labels=yte,average='macro')
    results.append([name,'ft',len(train),len(test),tt,t1,f1score,f'threshold={t}'])
    f1.append(f1score)
    preds.append(predicted)

  i = f1.index(max(f1))
  U.heatmap(f'../paper/images/{name}-ft-{len(train)}-{len(test)}-{params[i]}.pdf', *U.confuse(yte,preds[i]))
  return results


def csv_format(setup,model,trainN,testN,trainT,testT,f1,notes):
  return f'{setup},{model},{trainN},{testN},{trainT:.2f},{testT:.2f},{f1:.3f},{notes}'


def runner(train:list,test:list,dest:Path,name:str):
  '''
  Given a train and test set, write CSV:
  model, train size, test size, train time, inference time, F1 score, hyperparams/notes
  '''
  with open(dest/f'{name}-result.csv','w') as f:
    print('setup,model,train(N),test(N),train(t),inference(t),F1,note',file=f)
    a,b = len(train),len(test)
    print(f'[{name}] train({a}) test({b})')
    # everything except pytorch
    if a>3200 or b>800:
      for s1,s2 in ((3200,800),(800,200)):
        for model,rnr in zip(('kn','rf','rt','ft'),(run_kn,run_rf,run_rt,run_ft)):
          print(model,s1,s2)
          for r in rnr(train[:s1],test[:s2],name): print(csv_format(*r),file=f)
    else:
      for model,rnr in zip(('kn','rf','rt','ft'),(run_kn,run_rf,run_rt,run_ft)):
        print(model,a,b)
        for r in rnr(train,test,name): print(csv_format(*r),file=f)



def main(args):
  '''
  1. choose subdirectory names under args.tmp and args.out
  2. build dataset of Obfuscated, Subset (of Obfuscated), and Plain files
  3. train models using Plain+Subset
  4. evaluate using Obfuscated
  '''
  objs_dir = args.tmp/'objs'
  strp_dir = args.tmp/'strip'
  obfus_dir = args.out/'o'    # object files: obfuscated
  plain_dir = args.out/'p'    # object files: plain

  if not (args.tmp/'make.txt').exists():
    print(f'Compiling and obfuscating in {args.tmp}.')
    tg.main(tg.parse_args((str(args.tmp),'musl-1.2.4','objs')))

  if not strp_dir.exists():
    print(f'Creating stripped object files in {strp_dir}.')
    strp_dir.mkdir(parents=True,exist_ok=True)
    orig,cull,kept = strip(objs_dir,strp_dir)
    print(f'Starting with {orig} object files, {cull} are larger than our size threshold ({MIN_OBJ_SIZE}). Of those, {kept} are actually usable')

  if not (obfus_dir.exists() and plain_dir.exists()):
    for x in (obfus_dir,plain_dir):
      x.mkdir(parents=True,exist_ok=True)
    print(f'Separating object files from {strp_dir} to into plain ({plain_dir}) and obfuscated ({obfus_dir}) directories.')
    for x in strp_dir.glob('*.o'):
      shutil.move(x, (plain_dir if x.stem.startswith('Plain-') else obfus_dir)/x.name)

  # run train/test on combinations of O/P
  otrain,otest = U.tvt(obfus_dir.glob('*.o'),2)
  ptrain,ptest = U.tvt(plain_dir.glob('*.o'),2)
  runner(otrain,otest,args.out,'oo') # O/O
  runner(otrain,ptest,args.out,'op') # O/P
  runner(ptrain,otest,args.out,'po') # P/O
  runner(ptrain,ptest,args.out,'pp') # P/P
  return

  # train with P+S, test with O or P
  # extract subset of functions from obfuscated/plain training sets
  subfuncs = 'abs asin atan2 daemon inet_addr isalnum memcmp strchr strcpy strstr strtok tan utime'.split()
  strain = [n for n in otrain+ptrain if U.fname(n) in subfuncs]
  random.shuffle(strain)
  runner(strain,otest,args.out,'so') # P+S/O
  runner(strain,ptest,args.out,'sp') # P+S/P


def parse_args(args):
 p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument('tmp', type=Path, help='build directory for generating dataset')
 p.add_argument('out', type=Path, help='results and figures will go here')
 return p.parse_args(args)


if __name__ == "__main__":
  main(parse_args(sys.argv[1:]))
