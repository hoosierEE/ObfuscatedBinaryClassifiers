'''
deflate+kNN inspired by https://aclanthology.org/2023.findings-acl.426.pdf
with improvements from https://kenschutte.com/gzip-knn-paper2/
'''
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from glob import glob
from itertools import permutations,repeat
from multiprocessing import cpu_count
from pathlib import Path
from tqdm import tqdm
from typing import Iterable,List
import argparse,gzip,random,sys,time,zlib
import matplotlib.pyplot as plt, seaborn as sns
import numpy as np, pandas as pd

NCD = lambda c1,c2,c12: (c12-min(c1,c2))/max(c1,c2)
seed = lambda: random.seed(42)


class Zlen:
 '''
 Source: https://github.com/kts/gzip-knn/blob/compute_ncd/gziplength.py

 More efficient way to calculate pairwise distances between data1 and
 (data2...dataN), because it avoids recalculating compress(data1).

 usage:
 z = Zlen(data1)
 n1 = z.l1             # size of compressed data1
 n2 = z.length(data2)  # len(compress(data1)) + len(compress(data2))
 n3 = z.length(data3)  # len(compress(data1)) + len(compress(data2))
 '''
 def __init__(self,data:bytes):
  c = zlib.compressobj(gzip._COMPRESS_LEVEL_BEST, zlib.DEFLATED, -zlib.MAX_WBITS, zlib.DEF_MEM_LEVEL, 0) # from gzip.py
  self.l0 = len(c.compress(data))         #initially zero
  self.c = c.copy()                       #snapshot after compressing first data
  self.l1 = self.l0 + len(c.flush())      #TYPICALLY USE THIS (length of compressed data)
  self.l2 = len(self.c.compress(b" "))    #compress a space

 def length(self,data:bytes):
  c = self.c.copy() #from snapshot
  return self.l0 + self.l2 + len(c.compress(data)) + len(c.flush())


def rank_ncd(x:bytes, train:Iterable[bytes], pre:Iterable[int]) -> 'np.float(len(traing))':
 D = np.zeros(len(train))
 g = Zlen(x)
 l1 = g.l1
 for i,t2 in enumerate(train):
  l2 = pre[i]
  l12 = g.length(t2)
  D[i] = NCD(l1,l2,l12)

 return np.argsort(D).astype(np.uint32)


def train_test_split(xs,fraction=0.9):
 frac = int(len(xs) * fraction)
 return xs[:frac], xs[frac:]


def read_bytes(xs:Path) -> List[bytes]:
 r = []
 for x in xs:
  with open(x,'rb') as f:
   r.append(f.read())

 return r


def do_knn_future(k, xtrain,xtest,ytrain,ytest) -> '(actual_labels, predicted_labels)':
 '''concurrent knn'''
 xtest_bytes = read_bytes(xtest)
 xtrain_bytes = read_bytes(xtrain)
 precomputed = [Zlen(x).l1 for x in xtrain_bytes]
 top_args = np.zeros((len(xtest),k or len(xtrain)),'uint32')
 with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
  executor.map(rank_ncd)
  futures = {
   executor.submit(rank_ncd, x, xtrain_bytes, precomputed):i
   for i,x in enumerate(xtest_bytes)
  }
  for future in as_completed(futures):
   i = futures[future]
   top_args[i] = future.result()[:k]

 lytrain = np.array(ytrain)
 lytest = np.array(ytest)
 pred = np.array([Counter(x).most_common(1)[0][0] for x in lytrain[top_args]])
 return (lytest,pred)


def precompute(binary):
 return Zlen(binary).l1


def do_nn_future(xtrain:Iterable[Path|bytes],xtest:Iterable[Path|bytes],as_bytes:bool=False) -> 'np.float(len(xtest),len(xtrain))':
 '''neighbor distances, ranked'''
 xtest_bytes = xtest if as_bytes else read_bytes(xtest)
 xtrain_bytes = xtrain if as_bytes else read_bytes(xtrain)
 with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
  precomputed = list(executor.map(precompute, xtrain_bytes, chunksize=100))
  return np.vstack(list(executor.map(
   rank_ncd, xtest_bytes, repeat(xtrain_bytes), repeat(precomputed), chunksize=10)))


def confuse(actual,predicted):
 act,pred = list(actual),list(predicted)
 f = sorted(set(act+pred))
 D = np.zeros((len(f),len(f)),'uint32')
 for a,p in zip(act,pred):
  D[f.index(a),f.index(p)] += 1

 return D,f


def heatmap(plotname, data, label):
 plt.clf();plt.close('all')
 df = pd.DataFrame(data, label, label)
 sns.set(font_scale=0.5)
 labels = [[x or '' for x in row] for row in data]
 ticks = np.arange(data.max()+1)
 ax = sns.heatmap(df, annot=labels, fmt="", cmap=sns.cm.mako_r, cbar=True, annot_kws={'size':5})
 ax.set(xlabel='predicted', ylabel='actual')
 # sns.heatmap(df, annot=False, fmt="", cmap=sns.light_palette('seagreen', as_cmap=True), cbar=True, annot_kws={'size':8})
 plt.savefig(plotname, dpi=500, bbox_inches='tight')


def test(sample=10):
 args = parse_args(f'../tig/data/plain --neighbors 3 --sample {sample}'.split())
 return main(args.objdir, args.sample)


def main(objdir:Path,sample:int):
 fcount = defaultdict(int)
 #TODO use Path.glob, no need for glob dependency.
 paths = [Path(x) for x in glob(str(objdir/'*.o'))]
 pp = defaultdict(list) #plain (TODO: currently unused)
 op = defaultdict(list) #obfuscated
 for p in paths:
  f = p.stem.split('-')[-1]
  if p.stem.startswith('Plain'):
   pp[f].append(p)
  else:
   op[f].append(p)

 # N random samples for each obfuscated function
 trn,tst = {},{}
 for f,ps in op.items():
  trn[f],tst[f] = train_test_split(random.sample(ps, min(sample,len(ps))), 0.8)

 # {func:[path]} => [(func,path)]
 tsts,trns = [],[]
 for f,ps in tst.items(): tsts.extend([(f,p) for p in ps])
 for f,ps in trn.items(): trns.extend([(f,p) for p in ps])
 # now we have our test and train data, let's train!
 ytrain,xtrain = zip(*trns)
 ytest,xtest = zip(*tsts)
 return np.array(ytrain),np.array(ytest),do_nn_future(xtrain,xtest)


def parse_args(args=sys.argv[1:]):
 p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument('objdir', type=Path, help='parent folder of all object files')
 p.add_argument('--neighbors', default=3, help='single int or comma-separated list e.g: --neighbors=3,5,9')
 p.add_argument('--sample', metavar='N', type=int, default=100, help='limit (N//len(classes)) per class')
 p.add_argument('--gridsearch', help='perform grid search across comma-separated parameters (e.g. --gridsearch=2,3,4)')
 return p.parse_args(args)


if __name__ == "__main__":
 args = parse_args()
 if args.gridsearch:
  gs = map(int,args.gridsearch.split(','))
  ks = map(int,args.neighbors.split(',')) # (3,5,9)
 else:
  gs = [args.sample]
  ks = [int(args.neighbors)]

 elapsed = 0
 print('k,sample,train samples,test samples,run time (seconds),precision,recall,accuracy')
 for s in gs:
  if Path(f'npy/top-{s}.npy').exists():
   predict = np.load(f'npy/predict-{s}.npy')
   actual = np.load(f'npy/actual-{s}.npy')
   top = np.load(f'npy/top-{s}.npy')
  else:
   start = time.time()
   predict,actual,top = main(args.objdir,s) #the slow part
   elapsed = time.time() - start
   np.save(f'npy/predict-{s}',predict)
   np.save(f'npy/actual-{s}',actual)
   np.save(f'npy/top-{s}',top)
  for k in ks:
   predicted = np.array([Counter(x).most_common(1)[0][0] for x in predict[top[:,:k]]]) #kNN
   acc = (actual==predicted).mean()
   matrix,labels = confuse(actual,predicted)
   diag = np.diag(matrix)
   precision = diag / matrix.sum(0)  # multiclass
   recall = diag / matrix.sum(1)  # multiclass
   accuracy = diag.sum() / matrix.sum() # aggregate for all classes
   print(f'{k},{s},{len(predict)},{actual.size},{elapsed:0.3f},{acc:0.3f},{precision.mean():0.3f},{recall.mean():0.3f},{accuracy:0.3f}')
   dname = str(args.objdir).split('/')[-1]
   name = f'{dname}_{s}_{k}'
   np.save(f'./npy/{name}_matrix',matrix)
   np.save(f'./npy/{name}_labels',labels)
   heatmap(f'../../paper/images/{name}.pdf',matrix,labels)
