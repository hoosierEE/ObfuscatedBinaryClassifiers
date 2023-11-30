import subprocess

def sh(cmd, *args, **kwargs):
  return subprocess.run(cmd, *args, encoding='utf-8', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs)
