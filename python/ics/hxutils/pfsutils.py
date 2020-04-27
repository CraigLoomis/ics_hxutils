import glob
import pathlib
import shlex
import subprocess
import time

import fitsio

def getCard(fname, card, hdu=0):
    hdr = fitsio.read_header(fname, hdu)
    return hdr[card]

def getCards(fname, cards, hdu=0):
    hdr = fitsio.read_header(fname, hdu)
    
    ret = []
    for c in cards:
        ret.append(hdr[c])
    return ret

def getTemps(fname, tempIds=(9,10,11)):
    return getCards(fname, [f'W_XTMP{i}' for i in tempIds])

def getRampInfo(fname):
    ret = getTemps(fname)
    ret.extend(getCards(fname, [f'W_XM1POS']))
    return ret
           
def oneCmd(actor, cmd, quiet=True, timeLim=10):
    t0 = time.time()
    
    logLevel = 'w' if quiet else 'i'
    args = shlex.split(f'oneCmd.py --level={logLevel} {actor} {cmd}')
    print(args)
    ret = subprocess.run(args, stderr=subprocess.PIPE, stdout=subprocess.PIPE,
                         encoding='latin-1')
    if not quiet:
        print(ret.stderr)
    ret.check_returncode()
    
    if not quiet:
        print(ret.stdout)
        
    t1 = time.time()
    if t1-t0 < 1.1:
        time.sleep(1.1-(t1-t0))
