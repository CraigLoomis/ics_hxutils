# This is junk for old JHU data

import glob
import os
import pathlib

import fitsio
import numpy as np

rootDir = "/data/ramps"
calibDir = "/data/pfsx/calib"
sitePrefix = "PFJB"
nightPattern = '20[12][0-9]-[01][0-9]-[0-3][0-9]'

def lastNight():
    nights = glob.glob(os.path.join(rootDir, nightPattern))
    nights.sort()
    return nights[-1]

def pathToVisit(path):
    path = pathlib.Path(path)
    return int(path.stem[4:-2], base=10)

def rampPath(visit=-1, cam=None, prefix=None):
    if prefix is None:
        prefix = sitePrefix
    if visit < 0:
        night = lastNight()
        fileGlob = '[0-9][0-9][0-9][0-9][0-9][0-9]'
    else:
        night = nightPattern
        fileGlob = '%06d' % visit

    if cam is None:
        fileGlob = f'{fileGlob}[0-9][0-9]'
    else:
        armNums = dict(b=1, r=2, n=3, m=4)

        # For b9/n9
        armNums = dict(b=3, n=3)
        fileGlob = '%s%d%d' % (fileGlob, int(cam[1]), armNums[cam[0]])

    ramps = glob.glob(os.path.join(rootDir, night, '%s%s.f*' % (prefix, fileGlob)))
    if visit < 0:
        return sorted(ramps)[visit]
    else:
        return ramps[0]

def lastRamp(prefix=None, cam=None):
    return rampPath(visit=-1, cam=cam, prefix=prefix)

def ramp(rampId, cam=None, prefix=None):
    """Given any sane id, return a FITS ramp.

    Args:
    rampId : int, or path, or ramp
      If already a ramp, return it.
      If a path, open and return the ramp
      If an int, treat as a visit, and resolve to a path using the cam.

    Returns:
    ramp : a fitsio FITS object
    """

    if isinstance(rampId, (int, np.integer)):
        pathOrFits = rampPath(rampId, cam=cam, prefix=prefix)
    else:
        pathOrFits = rampId

    if isinstance(pathOrFits, (str, pathlib.Path)):
        return fitsio.FITS(pathOrFits)
    else:
        return pathOrFits
