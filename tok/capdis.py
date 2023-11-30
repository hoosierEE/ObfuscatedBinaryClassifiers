'''Use Capstone to disassemble stripped object files.'''
from capstone import Cs,CsInsn,CS_ARCH_X86,CS_MODE_64
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import List
from _utils import fname
import re


def encode_hex(s:str) -> str:
 '''replace hex part of numeric constant with its length (also in hex)'''
 enc = lambda x:f'0x{len(x.group(1)):x}'
 return re.sub(r'0x([0-9a-fA-F]+)', enc, s)


def disasm_g(filename:Path) -> str:
 '''
 Like disasm, but separate punctuation tokens and compress hex literals:
 add byte ptr ds : [ rcx ] , al xor byte ptr [ rip + 0x8 ] , ch
 '''
 with open(filename,'rb') as f: code = f.read()
 return f'__label__{fname(filename)} {disasm_gs(code)}'


def disasm_gs(code:bytes) -> str:
 '''disassemble with finer granularity than `disasm()`'''
 md = Cs(CS_ARCH_X86,CS_MODE_64)
 x = ' '.join(f'{m} {o}' for (_,_,m,o) in md.disasm_lite(code,0))
 x = re.sub('([*,:+]|\[|\]|-(?!0))',r' \1 ',x)  #add spaces around punctuation
 x = re.sub(' +',' ',x)  #remove extra spaces to SAVE SPACE. lololol
 return encode_hex(x)


def disasm(filename:Path) -> str:
 '''
 Disassemble into 1 line of space-separated "tokens" starting with __label__<functionName>
 also compress hex literals (0x75627531 => 0x8)
 __label__abs add byte ptr ds:[rcx], al xor byte ptr [rip + 0x8], ch
 '''
 with open(filename,'rb') as f: code = f.read()
 return f'__label__{fname(filename)} {disasm_s(code)}'


def disasm_s(code:bytes) -> str:
 '''Disassemble a sequence of bytes into a space-separated string of tokens with `encode_hex()` literals.'''
 md = Cs(CS_ARCH_X86,CS_MODE_64)
 x = ' '.join(f'{m} {o}' for (_,_,m,o) in md.disasm_lite(code,0))
 return encode_hex(x)


def blocks(code:bytes) -> (List[bytes],List[str]):
 '''
 Split (code) into a list of its basic blocks.
 Return tuple of (byte format, string format).
 '''
 md = Cs(CS_ARCH_X86,CS_MODE_64)
 d = list(md.disasm(code,0))
 splits = [i for i,x in enumerate(d) if x.mnemonic=='endbr64']
 splits = list(zip([0]+splits,splits))
 b = [b''.join(x.bytes for x in d[i:j]) for i,j in splits]
 t = []
 for i,j in splits:
  s = []
  for x in d[i:j]: s.extend([x.mnemonic,x.op_str])
  t.append(' '.join(s))
 return b,t


def disassemble(paths:List[Path],tokenizer=disasm) -> List[str]:
 '''Tokenize each object file in (paths) according to disassembly function (tokenizer).'''
 with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
  return list(executor.map(tokenizer, paths))
