'''
FastText for tokenized datasets.
Preprocess train and test data with capdis.py before use.
'''

from collections import defaultdict
from pathlib import Path
import argparse
import fasttext
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
import sys


def confuse(act,pred):
 '''confusion matrix'''
 f = sorted(set(act+pred))
 D = np.zeros((len(f),len(f)),'uint32')
 for a,p in zip(act,pred):
  D[f.index(a),f.index(p)] += 1

 return D,f


def heatmap(plotname, data, label) -> None:
 '''plot a heatmap in a particular style'''
 plt.clf();plt.close('all')
 df = pd.DataFrame(data, label, label)
 # sns.set(font_scale=0.5)
 fig,(ax1,ax2) = plt.subplots(1,2,figsize=(13,9), sharey=False, gridspec_kw={'width_ratios':[3,1]})
 annot = [[x or '' for x in row] for row in data]
 ticks = np.arange(data.max()+1)
 sns.heatmap(
  df,
  ax=ax1,
  linewidths=0.2, linecolor='white',
  cmap=sns.cm.mako_r, cbar=False, # cbar_kws={'boundaries':ticks},
  annot=annot, fmt="", annot_kws={'size':8})

 # heatmap
 ax1.set(xlabel='predicted', ylabel='actual')
 ax1.xaxis.set_label_position('top')
 ax1.xaxis.set_ticks_position('top')
 ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
 ax1.tick_params(left=False,top=False)

 # stacked bar showing relative [None]/correct/incorrect predictions
 ax2.tick_params(axis='y',which='both',left=False,labelleft=False)
 noguess = data[:,0]
 diag = np.diag(data)
 db = pd.DataFrame({
  '[None]': noguess,
  'right': diag,
  'wrong': data.sum(1) - (noguess+diag),
  'Label': label,
 })
 db.set_index('Label').plot(
  ax=ax2, kind='barh', stacked=True, color=['#bbb','#09a','#f33']
 ).legend(loc='lower right', bbox_to_anchor=(1,1))
 ax2.set_xticks([])
 ax2.get_yaxis().set_visible(False)
 sns.despine(ax=ax2, left=True,bottom=True)
 ax2.invert_yaxis()
 plt.subplots_adjust(wspace=0.01)
 plt.savefig(plotname, dpi=500, bbox_inches='tight')


def get_model(args) -> 'fasttext.model':
 '''load an existing model, or train a new one, or autotune a new one'''
 if args.model.exists():
  fasttext.FastText.eprint = lambda x: None
  model = fasttext.load_model(str(args.model))
 elif args.autotune:
  model = fasttext.train_supervised(input=str(args.train), autotuneValidationFile=str(args.test), autotuneDuration=args.autotune)
  model.save_model(str(args.model))
 else:
  model = fasttext.train_supervised(input=str(args.train), lr=0.01, epoch=10000, wordNgrams=1, loss='hs')
  model.save_model(str(args.model))

 return model


def predict(model,test,thresh,k):
 actual,predicted = [],[]
 with open(test) as f:
  for line in f:
   ln = line.strip()
   if ' ' in ln.strip():
    try:
     actual.append(ln[:ln.index(' ')][9:]) #get rid of __label__ prefix
     ks,_ = model.predict(ln,k=(k or len(model.labels)),threshold=thresh)
    except:
     print(line)
     exit()
    predicted.append(ks[0][9:] if ks else '[None]')

 return actual,predicted


def stats(matrix,labels):
 diag = np.diag(matrix)
 precision = diag / np.nan_to_num(matrix.sum(0))     # multiclass precision
 recall = diag / np.nan_to_num(matrix.sum(1))        # multiclass recall
 accuracy = diag.sum() / np.nan_to_num(matrix.sum()) # aggregate for all classes
 return labels,precision,recall,accuracy


def parse_args(args):
 p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument('train', type=Path, help='path to training data')
 p.add_argument('test', type=Path, help='path to test data')
 p.add_argument('--model', type=Path, default=Path('ft.bin'), help='name of model file to save/load')
 p.add_argument('--at', type=int, metavar='k', help='precision and recall @k [label size]')
 p.add_argument('--threshold', type=float, default=0.3, help='discard predictions below this level of confidence [0-1]')
 p.add_argument('--autotune', type=int, metavar='N', help='run fasttext autotune for N seconds')
 return p.parse_args(args)


if __name__ == "__main__":
 random.seed(0)
 args = parse_args(sys.argv[1:])
 assert 0 <= args.threshold <= 1
 model = get_model(args)
 if args.autotune:
  print(f'Autotuning complete. Model saved as {args.model}.')
  sys.exit(0)

 actual,predicted = predict(model,args.test,args.threshold,args.at)
 matrix,labels = confuse(actual,predicted)
 np.save('matrix.npy',matrix)
 np.save('labels.npy',labels)

 labels1,precision1,recall1,accuracy1 = stats(matrix,labels)
 labels2,precision2,recall2,accuracy2 = stats(matrix[1:,1:],labels[1:])
 testcount,traincount = defaultdict(int),defaultdict(int)
 with open(args.test) as f:
  for line in f.readlines():
   n = line[:line.index(' ')][9:]
   testcount[n] += 1
 with open(args.train) as f:
  for line in f.readlines():
   n = line[:line.index(' ')][9:]
   traincount[n] += 1

 tel,tec = zip(*sorted(testcount.items()))
 trl,trc = zip(*sorted(traincount.items()))
 at = args.at or 'all'
 hdr = f'|label       |train samples|test samples|p@{at:<5}|r@{at:<5}|*p@{at:<4}|*r@{at:<4}|'
 print(hdr)
 # print('|--'*hdr[1:].count('|')+'|')
 print( '|------------|-------------|------------|-------|-------|-------|-------|')
 for k,v1,v2,p,r,s,t in zip(trl,trc,tec,precision1,recall1,precision2,recall2):
  print(f'|{k:<12}|{v1:>13}|{v2:>12}|{p:>7.2}|{r:>7.2}|{s:>7.2}|{t:>7.2}|')

 v,w = np.array(list(trc)),np.array(list(tec))
 recall1,recall2 = np.nan_to_num(recall1),np.nan_to_num(recall2)
 print( '|------------|-------------|------------|-------|-------|-------|-------|')
 print(f'|mean        |{int(v.mean()):>13}|{int(w.mean()):>12}|{precision1.mean():>7.2}|{recall1.mean():>7.2}|{precision2.mean():>7.2}|{recall2.mean():>7.2}|')
 print(f'|std         |{int(v.std()):>13}|{int(w.std()):>12}|{precision1.std():>7.2}|{recall1.std():>7.2}|{precision2.std():>7.2}|{recall2.std():>7.2}|')

 heatmap(f'../../paper/images/fasttext-{args.model}.pdf',matrix,labels1)
 print(f'Average (multi-class) test accuracy: {accuracy1:.2%}')
 print(f'Average (multi-class) test accuracy: {accuracy2:.2%} (*discarding predictions below threshold={args.threshold})')
