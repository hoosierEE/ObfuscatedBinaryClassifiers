'''utilities for balancing and reading datasets'''

from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import cpu_count
from pathlib import Path
from tqdm import tqdm
from typing import Dict,List,Iterable,Tuple
import angr
import itertools
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import random
import re
import seaborn as sns
import subprocess as s
import walker # https://pypi.org/project/graph-walker/

# type aliases
Token,Index = str,int


def batched(iterable, n):
  "Batch data into tuples of length n. The last batch may be shorter."
  # source: https://docs.python.org/3.11/library/itertools.html
  # usage:  batched('ABCDEFG', 3) --> ABC DEF G
  if n < 1:
    raise ValueError('n must be at least one')
  it = iter(iterable)
  while batch := tuple(itertools.islice(it, n)):
    yield batch


def confuse(act,pred):
  '''confusion matrix'''
  f = sorted(set(act+pred))
  D = np.zeros((len(f),len(f)),np.uint32)
  for a,p in zip(act,pred): D[f.index(a),f.index(p)] += 1
  return D,f


def heatmap(plotname, data, label) -> None:
  # TODO: hbar should use colors from existing cmap
  '''plot a confusion matrix with summary along side'''
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
    cmap=sns.cm.mako_r, cbar=False,
    annot=annot, fmt="", annot_kws={'size':8}
  )

  # heatmap
  ax1.set(xlabel='predicted', ylabel='actual')
  ax1.xaxis.set_label_position('top')
  ax1.xaxis.set_ticks_position('top')
  ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
  ax1.tick_params(left=False,top=False)

  # stacked bar showing relative correct/incorrect/[None] predictions
  ax2.tick_params(axis='y',which='both',left=False,labelleft=False)
  diag = np.diag(data)
  certain = '[None]' not in label
  if certain:
    clrs = sns.cm.mako_r.resampled(3)((1,2))
    db = pd.DataFrame({
      'Correct': diag,
      'Incorrect': data.sum(1) - diag,
      'Label': label,
    })
  else:
    clrs = sns.cm.mako_r.resampled(3)((1,2,0))
    unk = data[:,0]
    db = pd.DataFrame({
      'Correct': diag,
      'Incorrect': data.sum(1) - (diag+unk),
      '[None]': unk,
      'Label': label,
    })

  db.set_index('Label').plot(
    ax=ax2, kind='barh', stacked=True, color=clrs
  ).legend(loc='lower right', bbox_to_anchor=(1,1))
  ax2.set_xticks([])
  ax2.get_yaxis().set_visible(False)
  sns.despine(ax=ax2,left=True,bottom=True)
  ax2.invert_yaxis()
  plt.subplots_adjust(wspace=0.01)
  plt.savefig(plotname, dpi=500, bbox_inches='tight')


def fname(p:Path) -> str:
  '''get the function name from "path/to/Transform-Option-name.o"'''
  return p.stem.split('-')[-1]


def sh(cmd, *args, **kwargs):
  '''run a shell command'''
  return s.run(cmd, *args, encoding='utf-8', shell=True, stdout=s.PIPE, stderr=s.STDOUT, **kwargs)


def get_binary(x:Path) -> 'np.uint32(256)':
  with open(x,'rb') as f: sample = f.read()
  x = np.zeros(256,dtype=np.uint32)
  for byte in sample: x[byte] += 1
  return x


def bag_of_bytes(paths:Iterable[Path]) -> 'np.uint32(len(paths),256)':
  '''Embed each file in (paths) into a 256-column vector of byte counts.'''
  with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
    return np.vstack(list(executor.map(get_binary, paths, chunksize=1000)))


def balance(paths:List[Path], limit:int=9999) -> Dict[str,List[Path]]:
  '''
  Balance dataset classes.
  Arbitrary default limit (9999) is orders of magnitude more than we need for our dataset.
  '''
  labels = [fname(x) for x in paths]
  vs = list(Counter(labels).values())
  limit = min(limit,int(np.mean(vs)+np.std(vs))) # min((user-defined limit),(1 stdev above mean))
  d = defaultdict(list)
  for k,v in zip(labels,paths): d[k].append(v) # gather paths by label
  return {k:v[:limit] for k,v in d.items()} # apply limit to number of items per class


def enlist(d:Dict[str,List[Path]]) -> List[Path]:
  '''Return flattened and shuffled dict values.'''
  p = []
  for k in d: p.extend(d[k])
  random.shuffle(p)
  return p


def make_vocab(corpus:str) -> Dict[Token,Index]:
  '''
  From a corpus of examples, split into tokens by whitespace,
  then id each token based on its frequency (descending).
  Return sorted and normalize dictionary with:
  - most frequent token id == 0
  - least frequent token id == len(set(words))
  - "not in corpus" => 0
  '''
  keys = (x for x in Counter(corpus.split()).keys() if not x.startswith('__label__'))
  return defaultdict(int,dict(map(lambda x:(x[1],x[0]),enumerate(keys,1))))


def tti(tokens:List[Token],vocab:Dict[Token,Index]) -> List[Index]:
  '''Map each token to its vocabulary index. Out-of-vocabulary (outdex) is None.'''
  return [vocab[t] for t in tokens]


def tvt(data:Iterable[Path],num_splits:int=2) -> List[List[Path]]:
  '''
  num_splits == 2 --> return 8:2   Train:Validation      splits from shuffled data
  num_splits == 3 --> return 8:1:1 Train:Validation:Test splits from shuffled data
  '''
  assert num_splits in (2,3), 'Only supporting train/test or train/val/test splits for now.'
  d = defaultdict(list)
  for x in data:
    d[fname(x)].append(x)

  d0,d1,d2 = [],[],[]
  for v in d.values():
    sz = len(v)
    a,b = int(sz*0.8),int(sz*0.9)
    d0.extend(v[:a])
    if num_splits == 3:
      d1.extend(v[a:b])
      d2.extend(v[b:])
    else:
      d1.extend(v[a:])

  random.shuffle(d0)
  random.shuffle(d1)
  random.shuffle(d2)
  return [d0,d1,d2][:num_splits]


def keep_classes(d:Dict[str,List[Path]], keep:list) -> Dict[str,List[Path]]:
  '''return a dataset with some classes missing'''
  return {k:v for k,v in d.items() if k in keep}


def blocks(path:Path) -> List[str]:
  '''instructions for all basic blocks in binary at (path).'''
  logging.getLogger('angr').setLevel(logging.ERROR) #default=logging.WARNING
  # https://docs.angr.io/en/latest/faq.html#why-is-angr-s-cfg-different-from-ida-s
  nodes = angr.Project(path).analyses.CFGFast(normalize=True).graph.nodes
  return [' '.join(' '.join(str(i).split()[1:])for i in n.block.capstone.insns)for n in nodes if n.block]


def insn_str(nodes:Iterable['nx.DiGraph.node']) -> List[str]:
  return [
    encode_punct(' '.join(' '.join(str(i).split()[1:]) for i in n.block.capstone.insns))
    for n in nodes if n.block
  ]


def encode_punct(x:str) -> str:
 '''replace hex part of numeric constant with its length (also in hex)'''
 x = re.sub('([*,:+]|\[|\]|-(?!0))',r' \1 ',x)  #add spaces around punctuation
 x = re.sub(' +',' ',x)  #remove extra spaces to SAVE SPACE. lololol
 enc = lambda x:f'0x{len(x.group(1)):x}'
 return re.sub(r'0x([0-9a-fA-F]+)', enc, x)


def rw(path) -> List[str]:
  '''random walks for object file at (path)'''
  logging.getLogger('angr').setLevel(logging.ERROR) #default=logging.WARNING
  project = angr.Project(path,load_options={'auto_load_libs':False})
  G = project.analyses.CFGFast(normalize=True).graph
  b = np.array(insn_str(G.nodes)) #blocks
  w = walker.random_walks(G,n_walks=4,walk_len=5,p=.25,q=.25,verbose=False) #node2vec values for p and q
  w = np.unique(w,axis=0) #no duplicate walks
  w = w[np.all(w<len(b),axis=1)] #discard out-of-bounds indexes (TODO: but why do they exist in the first place?)
  return path,b,w


def rwp(paths:List[Path]) -> List[Tuple[Path,List[str]]]:
  '''get random walks for every item in paths'''
  with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
    return list(executor.map(rw, paths, chunksize=max(1,len(paths)//cpu_count())))


def rw_dataset(
    prefix:Path,
    objdir:Path,
    N:int=400, #400*25 functions = 10000 examples
    merge_dashes=True # "blah-blah-func.o" => "func"
) -> Dict[str,List[Tuple[Path,List[str],'np.int(max(k,N),5)']]]:
  '''
  Create a dataset of basic blocks and random walks of those blocks for each binary.
  Save as .pkl files (one per funcname) e.g. prefix/abs.pkl, prefix/sin.pkl, etc.
  return Dict[funcname, {paths:List[Path], blocks:List[Block], walks:List[Walk]}]
  '''
  ps = list(objdir.glob('*.o'))
  random.shuffle(ps)
  fd = defaultdict(list)
  for p in ps: fd[fname(p) if merge_dashes else p.stem].append(p)
  dd = {}
  for k,v in tqdm(fd.items()):
    dd[k] = rwp(v[:N])
    with open(prefix/(k+'.pkl'),'wb') as f:
      pickle.dump(dd[k], f, protocol=pickle.HIGHEST_PROTOCOL)

  return dd


def read_pkl(path:Path) -> Tuple[Dict[str,Dict[str,List]], Counter]:
  '''
  Read ../out/b*/*pkl into one big dataset.
  Each function has a list: [(path, blocks, random walks), ...]
  Create a vocabulary of unique blocks and their counts to use as token indexes for embeddings.
  Return (data,vocab)
  '''
  vocab = Counter() #{block:count}
  data = defaultdict(dict)
  for p in path.glob('*.pkl'):
    print(p)
    with open(p, 'rb') as f:
      ds = pickle.load(f)

    paths,blocks,walks = zip(*ds)
    vocab.update(b for block in blocks for b in block)
    data[p.stem]['paths'] = paths
    data[p.stem]['blocks'] = blocks
    data[p.stem]['walks'] = walks

  return data,vocab
