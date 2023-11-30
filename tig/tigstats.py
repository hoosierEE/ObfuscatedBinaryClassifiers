'''
What functions were built correctly by mktig.py?
Print some stats and generate an image summary.
'''
from musl_tigress import FUNCS,OPTIONS,ORIGINAL_RECIPES
from collections import defaultdict
import numpy as np
import os
import matplotlib.pyplot as plt

funcs = [f'{x}.o' for x in FUNCS]
folders = os.listdir('out')
r = {k:np.zeros(len(funcs)) for k in ORIGINAL_RECIPES}

count = 0
objcount = 0
for subfolder in folders:
 objs = os.listdir('out/'+subfolder)
 objcount += len(objs)
 hits = np.array([int(y in objs) for y in funcs])
 count += 1
 for recip in ORIGINAL_RECIPES:
  if recip in subfolder:
   r[recip] += hits

print(f'{len(FUNCS)} functions')
print(f'{count} combinations of ({" ".join(ORIGINAL_RECIPES)}) and ({" ".join(list(OPTIONS.keys())[1:])})')
print(f'{objcount} total object files (about {objcount//len(FUNCS)} object files per function)')
print('about 402MB of object files on disk')

result = np.array(list(r.values()))
plt.imshow(result)
plt.xticks(np.arange(result.shape[1]), FUNCS, rotation=90)
plt.yticks(np.arange(result.shape[0]), ORIGINAL_RECIPES)
plt.savefig('a.pdf', dpi=500, bbox_inches='tight')
