#!/usr/bin/env python3
import subprocess
import sys

def run_script(name):
    print(f" Initiate Running {name} …")
    ret = subprocess.call([sys.executable, name])
    if ret != 0:
        print(f" Did not execute {name} exited with code {ret}")
        sys.exit(ret)
    print(f"✅ {name} finished successfully.")

def main():
    run_script('Full_UGV_Mission.py')
    run_script('dump_bed.py')

if __name__ == '__main__':
    main()