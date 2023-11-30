'''
Compile functions without obfuscation, but with diverse compilation options.
Useful for ablation studies.

The gen.py script in this folder produces object files named like so:

  dest/<Transforms>-<Options>-<filename>.o

Where <Transforms>, <Options>, and <filename> all vary.
We don't apply any Tigress tranformation, so the <Transforms> section is fixed to just be "Plain".
<Options> and <filename> are still variable, however:

  dest/Plain-<Options>-<filename>.o

Some concrete examples:

  dest/Plain-Loops-fread.o
  dest/Plain-Plain-fread.o
  dest/Plain-Safety-acos.o
'''


from _utils import sh
from gen import FUNCS,HEADERS,OPTIONS,ORIGINAL_RECIPES,setup,func_cmds

from glob import glob
from itertools import product
from pathlib import Path
from tqdm import tqdm
import argparse,os,sys


def parse_args(args=sys.argv[1:]):
 p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument('src', type=str, choices=['musl-1.2.4'], help='musl-version, i.e: musl-1.2.4')
 p.add_argument('tmp', type=Path, help='build directory for generating artifacts')
 p.add_argument('dst', type=Path, help='dataset directory name (will nest under tmp/)')
 return p.parse_args(args)


# def parse_args(args=sys.argv[1:]):
#  p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
#  p.add_argument('src', default='musl-1.2.4', help='musl source')
#  p.add_argument('--dst', default='noob-out', help='output directory name')
#  p.add_argument('--tmpdir', default='noob-tmp', help='add main to function and save here')
#  p.add_argument('--maketxt', default='make.txt', help='stdout from `make`')
#  p.add_argument('--dry-run', action='store_true', help='shell=print')
#  return p.parse_args(args)


def setup(args):
 '''
 Prepares the source code for the following Tigress step.
 1. download, unzip, and compile musl-1.2.4 (result: make.txt and a bunch of unused object files)
 2. make dest, tmp, and out directories
 3. extract compile commands from make.txt
 4. append main(){} to each function.c file
 5. remove -o <output.lo> from compile command
 return tuple of
 [0]: {function_name: [compile command as a list of strings]
 [1]:
 '''
 if not Path('musl-1.2.4').is_dir():
  sh('wget https://musl.libc.org/releases/musl-1.2.4.tar.gz')
  sh('tar xzf musl-1.2.4.tar.gz')
  sh('rm musl-1.2.4.tar.gz')

 if not Path(args.maketxt).exists():
  sh('musl-1.2.4/configure --disable-shared')
  sh(f'DESTDIR=dest make -j install > make.txt')

 sh(f'mkdir -p {args.dst}')
 sh(f'mkdir -p {args.tmpdir}')
 fcs = func_cmds(args)

 # append main() function to each file
 for cmd in fcs.values():
  filename = Path(cmd.split()[-1])
  name = Path(args.tmpdir) / Path(cmd.split()[-1]).name
  with open(filename) as f:
   txt = f.read()

  with open(name,'w') as g:
   print(txt, file=g)
   maintxt = 'int main(){return 0;}'
   if maintxt not in txt:
    print(maintxt, file=g)


 libcopies = set()
 r = {}
 for k,v in fcs.items():
  x = v.split()
  i = x.index('-o')
  r[k] = x[1:i]+x[i+2:] #remove -o <whatever>
  prefix = str(Path(r[k][-1]).parents[0])
  r[k] = ' '.join(r[k])
  for lib in glob(f'{prefix}/*.h'):
   if lib not in libcopies:
    sh(f'cp {lib} {args.tmpdir}')

   libcopies.add(lib)

 print(libcopies)
 print(len(libcopies))
 return r


if __name__ == "__main__":
 args = parse_args()
 if args.dry_run:
  sh = print

 fcs = setup(args)
 for o,f in tqdm(product(OPTIONS,fcs)):
  sh(f'mkdir -p {args.dst}/-{o}')
  cmd = f'gcc {fcs[f]} {OPTIONS[o]} -o {args.dst}/-{o}/{f}.o'
  ret = sh(cmd)
  if (not args.dry_run) and ret.returncode:
   print(ret.stdout)
   print(ret.stderr)

# log2f_data\.h casemap\.h getc\.h pow_data\.h nonspacing\.h exp_data\.h powf_data\.h wide\.h fdop\.h logf_data\.h punct\.h time_impl\.h putc\.h netlink\.h lookup\.h exp2f_data\.h sqrt_data\.h log_data\.h log2_data\.h alpha\.h __invtrigl\.h
