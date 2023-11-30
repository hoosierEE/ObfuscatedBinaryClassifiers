'''
Extracts a subset of functions from musl, obfuscates them, and compiles them
into a dataset of object files.

In more detail, this script does the following:
1. Download musl-1.2.4, patch it, make install > make.txt
2. parse make.txt, remove warnings, generate object file for each function
3. add main() (and other Tigress boilerplate) to .c file
4. create multiple different C sources using Tigress
5. compile object files into one big dataset (a directory of object files)

The name of the object file is its label with the format:
   TigressTransforms-CompileOptions-functionName.o
for example:
   FlattenSplitFlatten-Loops-abs.o
'''

from concurrent.futures import ThreadPoolExecutor,as_completed
from itertools import product
from multiprocessing import cpu_count
from pathlib import Path
from tqdm import tqdm
from typing import Dict,List
from _utils import sh
import argparse,os,sys, numpy as np


COMB_SIZE = 4  # max number of obfuscations applied

# FIXME: this is temporarily set to a small value for dependency testing
# COMB_SIZE = 1


FUNCS = (
 ' abs acos asin atan2 ceil cos exp floor sin tan'
 ' memccpy memcmp memmem wmemmove mmap munmap'
 ' inet_aton inet_addr getaddrinfo freeaddrinfo'
 ' isalnum stpcpy stpncpy strchr strcpy strncpy strstr strtok'
 ' fputc fputs fseek fopen fclose fread'
 ' sprintf snprintf'
 ' utime gettimeofday daemon waitpid'
 ' socket listen connect setsockopt'
).split()

RECIPES = {# <FUNC> will be replaced later
 'Flatten':    '--Transform=Flatten --Functions=<FUNC>',
 'Split':      '--Transform=Split --Functions=<FUNC>',
 'EncodeArith':'--Transform=EncodeArithmetic --Functions=<FUNC>',
 'RndArgs':    '--Transform=RndArgs --Functions=<FUNC>',
 'Inline':     '--Transform=Inline --InlineOptimizeKinds=constProp --Functions=<FUNC>',
 'Merge':      '--Transform=Merge --MergeOpaqueStructs=* --Functions=<FUNC>,init_tigress',
 'Branch':     '--Transform=InitBranchFuns --InitBranchFunsOpaqueStructs=list,array --InitBranchFunsCount=0?3 --Functions=<FUNC>',
}

TIGRESS_PREAMBLE = f'tigress --Seed=42 --Environment=x86_64:Linux:Gcc:4.6 <RECIPE> '
CLEANUP = ' --Transform=CleanUp --CleanUpKinds=annotations,fold,randomize,noMain,compress,removeLabelsAsValues '
KEYS = list(RECIPES.keys()) # NOTE: this MUST be a copy, not a reference
for ncombs in range(COMB_SIZE):
 for ks in filter(None,product(*([KEYS]*ncombs))):
  k = ''.join(ks)
  RECIPES[k] = ' '.join(RECIPES[ki] for ki in ks)

RECIPES['Plain'] = ' ' #add exactly one "empty" recipe
KEYS = list(RECIPES.keys()) # NOTE: overwrite it now, after finishing combinatorics
OPTIONS = {# Options we think are interesting:
 'Plain':' ',  #no extra options
 # https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#Optimize-Options
 'Loops':'-floop-parallelize-all -ftree-loop-if-convert -funroll-all-loops -fsplit-loops -funswitch-loops',
 # https://gcc.gnu.org/onlinedocs/gcc/Code-Gen-Options.html#Code-Gen-Options
 'Codegen':'-fstack-reuse=all -ftrapv -fpcc-struct-return -fcommon -fpic -fpie',
 # https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html#Instrumentation-Options
 'Safety':'-fsanitize=address -fsanitize=pointer-compare -fsanitize=address -fsanitize-address-use-after-scope',
 'O3':'-O3', # go brrrr
 'Os':'-Os', # smol
 # 'O0':'-O0', # how about no
}


def gen_maketxt(args) -> Path:
 '''download, patch, and compile musl to generate a make.txt file'''
 if not (args.tmp/args.src).is_dir(): #download and patch
  print(f'downloading musl to {args.tmp}')
  sh(f'wget -c https://musl.libc.org/releases/{args.src}.tar.gz -O - | tar xzf -', cwd=args.tmp)
  sh(f'patch -d {args.tmp} -s -p0 <{args.src}.patch') #apply patches

 if not (args.tmp/'make.txt').exists(): #compile
  print(f'compiling musl in {args.tmp}')
  sh(f'{args.src}/configure --disable-warnings --disable-shared', cwd=args.tmp)
  sh(f'DESTDIR=build make -j install > make.txt', cwd=args.tmp)

 return args.tmp / 'make.txt'


def get_cpath(maketext:Path) -> Dict[str,str]:
 '''get compiler commands for FUNCS from make.txt, return {func:path/to/source.c}'''
 outs = {}
 with open(maketext) as f:
  for x in f:
   x = ' '.join(y for y in x.strip().split() if not (y.startswith('-O') or y.startswith('-W'))) #remove warnings,opts
   if x.startswith('gcc') and x.endswith('.c') and ('malloc' not in x or 'mallocng' in x):
    cpath = x.split()[-1]
    func = cpath[:-2].split('/')[-1] # "path/to/func.c" => "func"
    if func in FUNCS:
     outs[func] = cpath

 return outs


def gen_templates(prefix:Path, options:str, funcs:dict) -> Dict[str,str]:
 '''Write function.c files with Tigress boilerplate.'''
 result = {}
 for i,(func,cpath) in enumerate(funcs.items()):
  with open(prefix/cpath) as f: txt = f.read()
  name = f'/tmp/tigress-input-{i:04}.c'
  result[func] = name
  with open(name, 'w') as g: #side effect - write preprocessed file to /tmp
   print(f'''
#include "tigress.h"
void megaInit(void);
void init_tigress(){{}}
{txt}
int main(){{megaInit(); init_tigress(); return 0;}}
''', file=g)

 return result


def oname(dest, recipe, option, func):
 '''construct filename in a custom format'''
 return f'{dest}/{recipe}-{option}-{func}'


def setup(args):
 '''Perform initial compilation and prepare directories for obfuscation step.'''
 prod = product(RECIPES,OPTIONS,FUNCS)
 if args.show:
  for rof in prod:
   print(oname(args.dst, *rof))
  sys.exit(0)

 os.makedirs(args.tmp/args.dst, exist_ok=True)
 maketext = gen_maketxt(args) #compile
 funcs = get_cpath(maketext) #extract C file path from make.txt
 gcc_opts = (
  ' -std=c99 -nostdinc -ffreestanding -D_XOPEN_SOURCE=700'
  ' -Imusl-1.2.4/arch/x86_64 -Imusl-1.2.4/arch/generic'
  ' -Iobj/src/internal -Imusl-1.2.4/src/include '
  ' -Imusl-1.2.4/src/internal -Iobj/include -Imusl-1.2.4/include -fPIC'
  ' -I/usr/local/bin/tigresspkg/3.3.3') # -c -o ../dst/out.o /tmp/in.c
 cfiles = gen_templates(args.tmp, gcc_opts, funcs)
 return gcc_opts,cfiles


def tigress(cwd,outfile,func,cpath,gcc_opts,recipe,option):
 '''
 Use a Tigress (recipe) to obfuscate a single source file (cpath) and save as (outfile).o
 Delete .c file when done, and delete .o file if there was a compile error.
 '''
 cmd = TIGRESS_PREAMBLE.replace('<RECIPE>',RECIPES[recipe]).replace('<FUNC>',func)
 cmd = f'{cmd} --out={outfile}.c {CLEANUP} {gcc_opts} -c -o {outfile}.o {OPTIONS[option]} {cpath}'
 r = sh(cmd,cwd=cwd)
 if r.returncode: Path(f'{cwd}/{outfile}.o').unlink(missing_ok=True)
 Path(f'{cwd}/{outfile}.c').unlink(missing_ok=True)
 return r


def parse_args(args=sys.argv[1:]):
 p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument('tmp', type=Path, help='build directory for generating artifacts')
 p.add_argument('src', type=str, choices=['musl-1.2.4'], help='musl-version')
 p.add_argument('dst', type=Path, help='dataset directory name (will nest under tmp/)')
 p.add_argument('--show', action='store_true', help='list files without actually creating them')
 p.add_argument('--run', nargs=2, help='run a particular command')
 return p.parse_args(args)


def main(args):
 '''
 1. make boilerplate func.c
 2. generate many obfuscated versions: Transform-Option-func.c
 3. compile each Transform-Option-func.c --> Transform-Option-func.o
 '''
 gcc_opts,cfiles = setup(args)
 if args.run:
  recipe,option,func = args.run[0].split('-')
  cpath = args.run[1]
  outfile = oname(args.dst,recipe,option,func)
  r = tigress(args.tmp,outfile,func,cpath,gcc_opts,recipe,option)
  print(r.stdout)
  sys.exit(r.returncode)

 with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
  futures = []
  lp = len(RECIPES)*len(OPTIONS)*len(cfiles)
  print(f'generating {lp} object files in {args.tmp}/{args.dst} (this may take a while...)')
  for func,cpath in tqdm(cfiles.items()):
   for recipe,option in product(RECIPES,OPTIONS):
   # for recipe,option in product(['Plain'],OPTIONS):  #only non-obfuscated functions
    oname_i = oname(args.dst,recipe,option,func)
    if not Path(oname_i+'.o').exists():
     futures.append(executor.submit(tigress,args.tmp,oname_i,func,cpath,gcc_opts,recipe,option))

   for future in as_completed(futures):
    try:
     _ = future.result()
    except KeyboardInterrupt:
     executor.shutdown(wait=False,cancel_futures=True)


if __name__ == "__main__":
 main(parse_args())
