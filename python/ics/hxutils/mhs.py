import shlex
import subprocess
import time

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

def takeRamp(cam, nread, nreset=1, exptype="dark", comment="", quiet=True):
    oneCmd(f'hx_{cam}', f'ramp nread={nread} nreset={nreset} exptype={exptype} objname=\"{comment}\"', quiet=quiet)

def darks(cam, nreps, nread, pause=10):
    for i in range(nreps):
        takeRamp(cam, nread, exptype="dark", comment=f"darks_{i+1}", quiet=False)
        if i < nreps-1:
            time.sleep(pause)
