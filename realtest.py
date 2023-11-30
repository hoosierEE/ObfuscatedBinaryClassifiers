from pathlib import Path
from tqdm import tqdm
import _utils as U
import pickle
import argparse


def main(prefix:Path=Path('test')):
  '''generate test binaries'''
  if not Path('static-binaries').exists():
    print('Clone this first: https://github.com/andrew-d/static-binaries')
    return

  print('binutils and misc others')
  FUNCS = '''ag ar heartbleeder ht ld lsciphers nano ncat nm nmap nping objcopy
  objdump python ranlib readelf socat strings yasm'''.split()
  paths = [Path('static-binaries/binaries/linux/x86_64')/f for f in FUNCS]
  for func in tqdm(paths):
    if (prefix/(func.stem+'.pkl')).exists():
      continue
    with open(prefix/(func.stem+'.pkl'),'wb') as f:
      pickle.dump([U.rw(func)],f,protocol=pickle.HIGHEST_PROTOCOL)

  print('mirai loader')
  if not (prefix/'mirai_loader.pkl').exists():
    mirai = Path('theZoo/malware/Source/Original/IoT.Mirai')
    if not mirai.exists():
      print('Clone this first: https://github.com/ytisf/theZoo')
      return

    mirai_loader = mirai/'loader/loader.strip'
    if not mirai_loader.exists():
      print('build mirai loader first and then strip symbols:')
      print(f'cd {mirai} && bash build.sh')
      print(f'strip -Xxs -o loader.strip loader')
      return

    with open(prefix/'mirai_loader.pkl','wb') as f:
      pickle.dump([U.rw(mirai_loader)],f,protocol=pickle.HIGHEST_PROTOCOL)

  print('wannacry')
  if not (prefix/'wannacry.pkl').exists():
    p = Path('theZoo/malware/Binaries/Ransomware.WannaCry')
    p = p/'ed01ebfbc9eb5bbea545af4d01bf5f1071661840480439c6e5babe8e080e41aa.exe'
    with open(prefix/'wannacry.pkl','wb') as f:
      pickle.dump([U.rw(p)],f,protocol=pickle.HIGHEST_PROTOCOL)

  print('cryptolocker')
  if not (prefix/'cryptolocker.pkl').exists():
    p = Path('theZoo/malware/Binaries/CryptoLocker_22Jan2014')
    p = p/'1002.exe' #NOTE: 1003.exe did not work
    with open(prefix/'cryptolocker.pkl','wb') as f:
      pickle.dump([U.rw(p)],f,protocol=pickle.HIGHEST_PROTOCOL)

  print('petya0')
  if not (prefix/'petya0.pkl').exists():
    p = Path('theZoo/malware/Binaries/Ransomware.Petya')
    p = p/'26b4699a7b9eeb16e76305d843d4ab05e94d43f3201436927e13b3ebafa90739.bin'
    with open(prefix/'petya0.pkl','wb') as f:
      pickle.dump([U.rw(p)],f,protocol=pickle.HIGHEST_PROTOCOL)

  print('petya1')
  if not (prefix/'petya1.pkl').exists():
    p = Path('theZoo/malware/Binaries/Ransomware.Petya')
    p = p/'4c1dc737915d76b7ce579abddaba74ead6fdb5b519a1ea45308b8c49b950655c.bin'
    with open(prefix/'petya1.pkl','wb') as f:
      pickle.dump([U.rw(p)],f,protocol=pickle.HIGHEST_PROTOCOL)

  # # mirai[0|1|2|3]
  # NOTE: none of these worked with angr
  # if not (prefix/'mirai0').exists():
  #   p = Path('theZoo/malware/Binaries/Linux.Mirai.B')
  #   qs = [
  #     # '03254e6240c35f7d787ca5175ffc36818185e62bdfc4d88d5b342451a747156d',
  #     # 'acb930a41abdc4b055e2e3806aad85068be8d85e0e0610be35e784bfd7cf5b0e',
  #     # 'f60b29cfb7eab3aeb391f46e94d4d8efadde5498583a2f5c71bd8212d8ae92da',
  #     # 'fcf603f5d5e788c21acd4a1c7b36d6bc8f980b42cf1ef3f88e89735512637c24'
  #   ]
  #   for i,q in enumerate(qs):
  #     print(p/q)
  #     with open(prefix/f'mirai{i}','wb') as f:
  #       pickle.dump(U.rw(p/q),f,protocol=pickle.HIGHEST_PROTOCOL)

  # # wannacry_plus
  # NOTE: none of these worked with angr
  # if not (prefix/'wannacry_plus').exists():
  #   p = Path('theZoo/malware/Binaries/Ransomware.WannaCry_Plus')
  #   p = p/'Ransomware.WannaCry_Plus.md5'
  #   with open(prefix/'wannacry','wb') as f:
  #     pickle.dump(U.rw(p),f,protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument('dest', type=Path)
  args = p.parse_args()
  main(args.dest)
