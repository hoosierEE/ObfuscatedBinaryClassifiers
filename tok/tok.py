'''NOTE: deprecated by capdis.py'''
from capstone import Cs,CS_ARCH_X86,CS_MODE_64
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
import random
import sys

random.seed(42)

def disassemble_one(f:Path) -> str:
 with open(f,'rb') as ff: code = ff.read()
 md = Cs(CS_ARCH_X86,CS_MODE_64)
 tokens = ' '.join(f'{mnemonic} {opstr}' for (_,_,mnemonic,opstr) in md.disasm_lite(code,0))
 return f.stem,tokens


def read_data(objs):
 fs = defaultdict(list) #separate by function
 print('disassembling object files...')
 with ThreadPoolExecutor(max_workers=17) as executor:
  futures = []
  for f in objs:
   futures.append(executor.submit(disassemble_one, f))

  for future in tqdm(as_completed(futures)):
   try:
    lbl,text = future.result()
    fs[lbl].append(text)
   except KeyboardInterrupt:
    executor.shutdown(wait=False,cancel_futures=True)

 return fs


def write_file(filename,data,kind,N=None):
 idx = {'all':None, 'transform':0, 'option':1, 'function':2,}[kind]
 with open(str(filename)+'.'+kind,'w') as f:
  for key in tqdm(data):
   docs = data[key][:N] #copy because shuffle is in-place. Also take at most N
   if idx is None: lbl = ' '.join(f'__label__{ki}' for ki in str(key).split('-'))
   else: lbl = f"__label__{str(key).split('-')[idx]}"
   for x in docs[:N]:
    y = x.replace(',',' ,').replace('[', '[ ').replace(']', ' ]')
    print(lbl,y, file=f)


def parse_args(args):
 p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument('src', type=Path, help='directory containing object files')
 p.add_argument('out', type=Path, help='name for generated training file')
 p.add_argument('--kind', choices=['all','transform','option','function'], default='function', help='label type')
 p.add_argument('--limit', type=int, help='max number of samples per class')
 return p.parse_args(args)


if __name__ == "__main__":
 args = parse_args(sys.argv[1:])
 objs = list(args.src.glob('*.o'))
 random.shuffle(objs)
 fs = read_data(objs)
 write_file(args.out, fs, args.kind, args.limit)
