# FIXME: where do extract_feats and get_binaries come from???
# They've been moved to ../_utils.py but not updated in here
# TODO: fix imports
from collections import defaultdict
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict,List,Iterable
import numpy as np
import random
import re


random.seed(42)


def get_stats(model, data, test, labels, description):
 X,y = get_binaries(data)
 Xf = extract_feat(X)
 model.fit(Xf,y)  #train
 Xt,ytest = get_binaries(test)
 Xtf = extract_feat(Xt)
 pred = model.predict(Xtf)
 accuracy = accuracy_score(ytest,pred)
 precision = precision_score(ytest,pred, average='macro', labels=labels, zero_division=0)
 recall = recall_score(ytest,pred, average='macro', labels=labels, zero_division=0)
 f1 = f1_score(ytest,pred, average='macro', labels=labels, zero_division=0)
 given,predict = description.split(':')
 stats = (f'|  {given:<5} {"("+str(len(y))+")":>7} | {predict:<5} {"("+str(len(ytest))+")":>6} | {accuracy:.2f} | {precision:.2f} | {recall:.2f} | {f1:.2f} |')
 return model,stats


def example(train:List[Path],test:List[Path],ytrain:List[str],ytest:List[str]) -> [str,float,float,float,int]:
 model = RandomForestClassifier(n_estimators=40, n_jobs=-1)
 X,y = get_binaries()


def toy():
 plain,ob = plain_split(Path('../tig/data/strip').glob('*.o'))
 b = balance(ob,48)
 labels = list(b.keys())
 data = enlist(b)
 split = int(len(data)*0.2)
 test,train = data[:split],data[split:]
 model = RandomForestClassifier(n_estimators=40, n_jobs=-1)
 X,y = get_binaries(data)
 Xf = extract_feat(X)
 model.fit(Xf,y)  #train
 Xt,ytest = get_binaries(test)
 Xtf = extract_feat(Xt)
 return model,Xtf,ytest


def table(estimators, limit, partial=False):
 plain,ob = plain_split(Path('../tig/data/strip').glob('*.o'))
 b = balance(ob,limit)
 labels = list(b.keys())
 data = enlist(b)
 split1 = int(len(data)*0.2)
 test,train = data[:split1],data[split1:]
 lshuf = labels[:]
 random.shuffle(lshuf)
 part1 = enlist(discard_classes(b, lshuf[int(len(lshuf)*1/3):]))  #discard 1/3 of the classes
 part2 = enlist(discard_classes(b, lshuf[int(len(lshuf)*1/2):]))  #discard half of the classes
 part3 = enlist(discard_classes(b, lshuf[int(len(lshuf)*2/3):]))  #discard 2/3 of the classes
 split2 = int(len(plain)*0.2)
 ptest,ptrain = plain[:split2],plain[split2:]
 header = r'| train | \norm{train} | test | \norm{test} | accuracy | precision | recall | f1 |'  #contains LaTeX
 print(header)
 print(re.sub('[^|]','-',header))
 model = lambda: RandomForestClassifier(n_estimators=estimators, class_weight='balanced', n_jobs=-1)  #create a fresh model each time
 if partial:
  m,s = get_stats(model(), part1,  test, labels, 'S_{a}:all')
  print(s)
  print(get_stats(model(), part2,  test, labels, 'S_{b}:all')[1])
  print(get_stats(model(), part3,  test, labels, 'S_{c}:all')[1])
  print(get_stats(model(), part1, plain, labels, 'S_{a}:plain')[1])
  print(get_stats(model(), part2, plain, labels, 'S_{b}:plain')[1])
  print(get_stats(model(), part3, plain, labels, 'S_{c}:plain')[1])
 else:
  m,s = get_stats(model(), train,   test, labels, 'all:all')
  print(s)
  print(get_stats(model(), train,  plain, labels, 'all:plain')[1])
  print(get_stats(model(), plain,   test, labels, 'plain:all')[1])
  print(get_stats(model(), ptrain, ptest, labels, 'plain:plain')[1])

 return m


# def thresh(model,Xtr,ytest,threshold):
#  '''idea: report top N with probabilities, so we can threshold'''
#  p = model.predict_proba(Xtr)
#  pp = p[:,-5:]
#  top5 = np.argsort(p, axis=1)[:,-5:]


if __name__ == "__main__":
 model = table(40,46)
 # model,Xtf = toy()
 # p = model.predict_proba(Xtf)
 # p>p.mean(0)
