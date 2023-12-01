'''
Basic block (plus random walks) embedding classifier.
Originally inspired by:
https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

Uses a siamese-ish network to learn blocks+walks simultaneously.
Trains on concat(plain,(subset of obfuscated)) examples.
Gets ~0.95 accuracy in ~4 seconds of triaining on single GPU.
'''

from collections import Counter,defaultdict
from pathlib import Path
from itertools import islice
from torch.utils.data import DataLoader,Dataset,random_split
from tqdm import tqdm
from typing import List,Tuple
import argparse
import numpy as np
import numpy.ma as ma
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = torch.Generator().manual_seed(42)

LABELS = {x:i for i,x in enumerate('''[UNK] abs acos asin atan2 ceil cos
 daemon floor inet_addr inet_aton isalnum memccpy memcmp memmem sin stpcpy
 stpncpy strchr strcpy strncpy strstr strtok tan utime wmemmove'''.split())}
# LABELS = {x:i for i,x in enumerate('''abs acos asin atan2 ceil cos
#  daemon floor inet_addr inet_aton isalnum memccpy memcmp memmem sin stpcpy
#  stpncpy strchr strcpy strncpy strstr strtok tan utime wmemmove'''.split())}
SLEBAL = {v:k for k,v in LABELS.items()} #reverse lookup


def read_pkl(paths:List[Path]) -> Tuple[dict,dict]:
  '''
  Read ../out/bo/*pkl into one big dataset.
  Create a vocabulary of unique blocks and their counts.
  Use that vocab to create block embeddings for functions (and their random walks).
  Also build a dataset
  Return (data,vocab)
  '''
  vocab = Counter()
  data = defaultdict(dict)
  for p in tqdm(paths, desc=f'reading .pkl files', total=len(paths)):
    with open(p,'rb') as f: ds = pickle.load(f)
    paths,blocks,walks = zip(*ds)
    vocab.update(b for block in blocks for b in block)
    data[p.stem]['paths'] = paths
    data[p.stem]['blocks'] = blocks
    data[p.stem]['walks'] = walks

  return dict(data),vocab


class BlockDataset(Dataset):
  ''' Big list of (label_index, List[Blocks]). '''
  def __init__(self,data:dict,vocab:dict):
    xs = [] # (label,block_indexes,walks)
    for k,v in tqdm(data.items(),desc='BlockDataset'):
      _,vb,vw = v.values()
      for b,w in zip(vb,vw):
        tok = np.array([vocab[x] for x in b])
        label = LABELS[k] if k in LABELS else LABELS['[UNK]'] #dummy value for inference
        bs = tok[w] # np.random.permutation(tok[w]) # .ravel() # [:len(tok)] #less than full b[w]
        xs.append((label, tok, bs)) # NOTE: replacing walk indexes with vocab indexes here

    np.random.shuffle(xs)
    self.data = xs

  def __len__(self): return len(self.data)
  def __getitem__(self,i): return self.data[i]


class BlockClassifier(nn.Module):
  def __init__(self, nclasses, vocab_size, e0_dim, e1_dim, pool_dim):
    super(BlockClassifier, self).__init__()
    self.e0 = nn.EmbeddingBag(vocab_size,e0_dim,mode='max')
    self.e1 = nn.EmbeddingBag(vocab_size,e1_dim,mode='max')
    self.drop = nn.Dropout(p=0.45)
    self.nonl = nn.LeakyReLU()
    self.pool = nn.AdaptiveMaxPool1d(pool_dim)
    self.fc = nn.Linear(pool_dim,nclasses) #output fc layer
    i = 1e-1
    self.e0.weight.data.uniform_(-i,i)
    self.e1.weight.data.uniform_(-i,i)
    self.fc.weight.data.uniform_(-i,i)
    self.fc.bias.data.zero_()

  def forward(self,tokens,walks,ot,ow):
    x = torch.cat((self.e0(tokens,ot),self.e1(walks,ow)), dim=1)
    # x = self.e0(tokens,ot) + self.e1(walks,ow)  #only works if same size
    x = self.drop(x)
    x = self.pool(x)
    x = self.nonl(x)
    x = self.drop(x)
    x = self.fc(x)
    return x


def train(model,optimizer,criterion,loader0,loader1):
  '''
  pseudo-siamese training loop:
  train on one dataset, then another, add losses and *then* backprop.
  '''
  model.train()
  acc,count = 0,0
  if loader1:
    alpha = 0.5 # (alpha)*loss0 + (1-alpha)*loss1
    for (l0,*data0),(l1,*data1) in zip(loader0,loader1):
      optimizer.zero_grad()
      pred0 = model(*data0)
      pred1 = model(*data1)
      loss = alpha * criterion(pred0,l0)
      loss += (1-alpha) * criterion(pred1,l1)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(),0.1)
      optimizer.step()
      acc += (pred0.argmax(1) == l0).sum().item()
      acc += (pred1.argmax(1) == l1).sum().item()
      count += l0.size(0) + l1.size(0)

  else:
    for (l0,*data0) in loader0:
      optimizer.zero_grad()
      pred0 = model(*data0)
      loss = criterion(pred0,l0)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(),0.1)
      optimizer.step()
      acc += (pred0.argmax(1) == l0).sum().item()
      count += l0.size(0)

  return loss.item(),acc/count


def evaluate(model,loader,confusion=None):
  model.eval()
  acc,counts,stacks = 0,0,[]
  with torch.no_grad():
    for labels,*data in loader:
      pred = model(*data)
      if confusion is not None:
        np.add.at(confusion,(pred.argmax(1).cpu(),labels.cpu()),1)

      acc += (pred.argmax(1) == labels).sum().item()
      counts += labels.size(0)

  return acc/counts


  #return (ke ,vocab,odata,otst,pdata,sdata)
def train_main(vocab,train0,train1,val,test,prefix=Path('.'),EPOCHS=150,mo=0.5,LR=20) -> ('model',dict):
  #hyperparameters
  E0_DIM = 200 # blocks
  E1_DIM = 80 # walks
  POOL_DIM = 80 # pooling layer size (right before FC output layer)
  t0 = time.time()
  # MODEL
  model = BlockClassifier(len(LABELS),len(vocab),E0_DIM,E1_DIM,POOL_DIM).to(device)
  # training loop
  optimizer = torch.optim.SGD(model.parameters(),lr=LR,momentum=mo)
  criterion = torch.nn.CrossEntropyLoss()
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.9,patience=10,verbose=1)
  acc,epoch,accs = 0,0,[]
  while epoch<EPOCHS:
    t1 = time.time()
    loss,acc_trn = train(model,optimizer,criterion,train0,train1)
    acc_val = evaluate(model,val)
    accs.append((loss,acc_trn,acc_val))
    if acc>acc_val: scheduler.step(loss)
    else:           acc = acc_val
    print(f'epoch:{epoch:>2} ({time.time()-t1:.2f}s) | train:{acc_trn:.2f}  val:{acc_val:.2f}')
    epoch += 1

  t0 = time.time() - t0 # elapased

  # evaluation and confusion matrix for heatmaps
  confusion = np.zeros((len(LABELS),len(LABELS)), dtype=int)
  t1 = time.time()
  for _ in range(10):
    evaluate(model,test,confusion)
  t1 = time.time() - t1

  # Compute:
  # - Per-class F1 score
  # - F1 macro
  # - F1 micro
  confusion = confusion[1:,1:] #don't count [UNK] class
  TP = np.diag(confusion) # true positives
  pre = np.nan_to_num(TP/confusion.sum(1),0) # per-class precision
  rec = np.nan_to_num(TP/confusion.sum(0),0) # per-class recall
  F1 = np.nan_to_num((2*pre*rec)/(pre+rec),0)
  f1dict = {SLEBAL[i]:x for i,x in enumerate(F1,start=1)}
  print('F1 score per class:')
  for i,x in enumerate(F1,start=1):
    print(f'{x:.3f}')
    # print(f'{SLEBAL[i]:<11} {x:.3f}')

  print('-----')
  print(f'{F1.mean():.3f}')
  print(f'{TP.sum()/confusion.sum():.3f}')
  # print(f'F1 (macro): {F1.mean():.3f}')
  # print(f'F1 (micro): {TP.sum()/confusion.sum():.3f}')
  # print(model)
  print(f'Total params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}. Train ({t0:.3f}s) Test ({t1:.3f})')

  # save model and stats
  torch.save(model,prefix/'model.pt')
  stats = {'confusion':confusion,'f1':f1dict,'training':accs}
  with open(prefix/'stats.pkl','wb') as f:
    pickle.dump(stats,f,protocol=pickle.HIGHEST_PROTOCOL)

  return model,stats


def collate_batch(batch) -> '5-tuple':
  '''
  Input (batch) is a batch-sized list of 3-tuples: (label_index, Tensor[Block(int)], Tensor[Walk(int)]).
  Appends a pair of offsets:
  - one to indicate where each group of tokens starts
  - another to indicate where each group of walks starts
  '''
  labels,tokens,walks,ot,ow = [],[],[],[0],[0]
  T = torch.int64
  for l,t,w in batch:
    labels.append(l)
    tokens.append(torch.tensor(t,dtype=T))
    wlks = torch.tensor(w.ravel(),dtype=T)
    walks.append(wlks)
    ot.append(len(t))
    ow.append(wlks.size(0))

  labels = torch.tensor(labels,dtype=T)
  tokens = torch.cat(tokens)
  walks = torch.cat(walks)
  ot = torch.tensor(ot[:-1]).cumsum(dim=0)
  ow = torch.tensor(ow[:-1]).cumsum(dim=0)
  return (labels.to(device),tokens.to(device),walks.to(device),ot.to(device),ow.to(device))


class Vocab(dict):
  '''Like defaultdict, but missing keys get the value of len(dict)+1.'''
  def __init__(self,*args,**kwargs):
    super(Vocab,self).__init__(*args,**kwargs)
    self.sz = len(self)
  def __missing__(self,x):
    self.sz += 1
    self[x] = self.sz
    return self.sz


def train_prep(prefix:Path=Path('.'),keep=0.5):
  odata,ovocab = read_pkl(sorted((prefix/'bo').glob('*.pkl')))
  pdata,pvocab = read_pkl(sorted((prefix/'bp').glob('*.pkl')))
  unk = int('[UNK]' in LABELS)
  num_classes = len(LABELS)
  subkeys = (unk+torch.randperm(num_classes-unk,generator=generator))[:int(num_classes*keep)]
  keeps = [SLEBAL[x.item()] for x in subkeys]
  sdata = {k:odata[k] for k in keeps}
  _vocab = Counter()
  _vocab.update(ovocab)
  _vocab.update(pvocab)
  _v,_ = zip(*_vocab.most_common()) #discard actual counts - we care more about ordinal rank
  vocab = Vocab(map(reversed,enumerate(_v,start=1)))
  with open(prefix/'vocab.pkl','wb') as f:
    pickle.dump(vocab,f,protocol=pickle.HIGHEST_PROTOCOL)

  # we use (ov) for validation and (ot) for testing
  bdo = BlockDataset(odata,vocab)
  bdo.data = bdo.data[:1200]
  bdp = BlockDataset(pdata,vocab)
  bds = BlockDataset(sdata,vocab)
  bds.data = bds.data[:1000]
  (ot,ov,oe) = (DataLoader(x,batch_size=100,shuffle=1,collate_fn=collate_batch) for x in random_split(bdo,[.8,.1,.1],generator=generator))
  (pt,pv,pe) = (DataLoader(x,batch_size=32,shuffle=1,collate_fn=collate_batch) for x in random_split(bdp,[.8,.1,.1],generator=generator))
  (st,sv) = (DataLoader(x,batch_size=100,shuffle=1,collate_fn=collate_batch) for x in random_split(bds,[.8,.2],generator=generator))
  return (keeps,vocab,ot,ov,oe,pt,pv,pe,st,sv)


def batched(iterable, n):
  '''
  Batch data into tuples of length n.
  The last batch may be shorter.
    batched('ABCDEFG', 3) --> ABC DEF G
  '''
  if n < 1:
    raise ValueError('n must be at least one')
  it = iter(iterable)
  while batch := tuple(islice(it, n)):
    yield batch


def collate_program(blocksize,walksize):
  def _f(batch) -> '5-tuple':
    '''
    Unlike collate_batch, which considers many small functions, in
    collate_program, we are primarily concerned with chunking a large program into
    many function-sized parts.  In particular, each (l,t,w) tuple in the batch
    will contain not multiple (l)s, (t)s, and (w)s, but more likely just one of
    each - and the (t) and (w) will be HUGE compared to what collate_batch deals with.

    So we're going to chunk each (t) and (w) into smaller bits here.  And at the
    end, we'll end up with List[List[Block]] and List[List[Walk]] rather than
    List[Block] and List[Walk]. Similarly we'll create additional
    position-encoding tensors so our model knows where each chopped-up block or
    walk starts.
    '''
    labels,tokens,walks,ot,ow = [],[],[],[0],[0]
    T = torch.int64
    for l,t,w in batch:
      for tb,wb in zip(batched(t,blocksize),batched(w,walksize)):
        wb = np.array(wb)
        rows,cols = wb.shape
        labels.append(l) #replicate label
        toks = torch.tensor(tb,dtype=T)
        tokens.append(toks)
        # old version:
        m = np.ma.masked_array(wb, np.c_[np.zeros_like(wb[:,0]), wb[:,1:]==wb[:,:-1]]).compressed()
        wlks = torch.tensor(m,dtype=T)
        walks.append(wlks)
        ow.append(wlks.size(0))

        #TODO test me
        # m = np.concatenate([np.zeros((rows,1)),wb[:,1:]==wb[:,:-1]],axis=1)
        # m = np.ma.masked_array(wb,m)
        # walks.append(torch.tensor(m.compressed(),dtype=T))
        # ow.extend(cols-m.mask.sum(1))

        ot.append(toks.size(0))

    labels = torch.tensor(labels,dtype=T)
    tokens = torch.cat(tokens)
    walks = torch.cat(walks)
    ot = torch.tensor(ot[:-1]).cumsum(dim=0)
    ow = torch.tensor(ow[:-1]).cumsum(dim=0)
    return (labels.to(device),tokens.to(device),walks.to(device),ot.to(device),ow.to(device))
  return _f


def test_preds(vocab,prefix:Path,paths:List[Path],step:int=20,chunk:int=30,threshold:float=0.8):
  '''
  Load vocab and model saved at (prefix) from earlier training, then turn (paths) into a dataset.
  Slice each file in the dataset into (chunk)-sized pieces, and scan in (step)-sized increments.
  Keep predictions higher than (threshold), otherwise [UNK].
  '''
  model = torch.load(prefix/'best_model.pt')
  model.eval()
  logits,predictions = [],[]
  # test dataset
  tdata,tvocab = read_pkl(paths)
  vocab.update(tvocab)
  ds = BlockDataset(tdata,vocab)
  td = DataLoader(ds,collate_fn=collate_program(chunk,chunk*2))
  # vote for top prediction in (step)-sized sliding windows
  for test in tqdm(td):
    pred = torch.nn.Softmax(1)(model(*test[1:])).cpu().detach().numpy()
    logits.append(pred)
    pred = (pred*(pred>threshold)).argmax(1)
    predictions.append(np.array([Counter(x).most_common(1)[0][0].item() for x in batched(pred,step)]))

  return (logits,predictions)


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument('train',type=Path,help='parent directory of out/bo/*pkl and out/bp/*pkl files for training')
  p.add_argument('test',type=Path,help='parent directory of *.pkl and *.pt files for testing')
  return p.parse_args()


# TODO: "no module named tch.block" error
if __name__ == "__main__":
  args = parse_args()
  print('main')
  if not ((args.train/'best_model.pt').exists() and (args.train/'vocab.pkl').exists()):
    print('Training a new model...')
    keeps,vocab,*ds = train_prep(args.train)
    with open(args.train/'vocab.pkl','wb') as f: pickle.dump(vocab,f,protocol=pickle.HIGHEST_PROTOCOL)
    _ = train_main(vocab,*ds,prefix=args.train)
    print(f'Subset includes: {" ".join(keeps)}')

  with open(args.train/'vocab.pkl','rb') as f:
    vocab = pickle.load(f)

  paths = sorted(args.test.glob('*.pkl'))
  predictions = []
  for pbatch in tqdm(batched(paths,5)):
    predictions.extend(test_preds(args.test,paths)[1])

  print(f'Saving predictions in {args.test}/predictions.pkl')
  with open(args.test/'predictions.pkl','wb') as f:
    pickle.dump(dict(zip((p.stem for p in paths),predictions)), f, protocol=pickle.HIGHEST_PROTOCOL)
