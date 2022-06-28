# This is junk for old JHU data

import glob
import logging
import os
import pathlib

import fitsio
import numpy as np

from . import hxramp

logger = logging.getLogger('')
logger.setLevel(logging.INFO)

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

def getPathsBetweenVisits(visit0, visitN=None, dateGlob='2022*', cam='n3'):
    if visitN is not None and visitN < visit0:
        raise ValueError('valueN must not be smaller than visit0')

    fileGlob = f'{sitePrefix}*{cam[-1]}3.fits'
    ramps = glob.glob(os.path.join(rootDir, dateGlob, fileGlob))
    ramps =  sorted(ramps)

    # import pdb; pdb.set_trace()
    visit0Pat = f'{sitePrefix}{visit0:06d}{cam[-1]}3.fits'
    logger.warning(f'checking {len(ramps)} against {visit0Pat}')
    if visitN is not None:
        visitNPat = f'{sitePrefix}{visitN:06d}{cam[-1]}3.fits'
    startAt = None
    for p_i, p in enumerate(ramps):
        if p.endswith(visit0Pat):
            logger.warning(f'matched start {p} against {visit0Pat}')
            startAt = p_i
            break
    if startAt is None:
        return []

    if visitN is None:
        return ramps[startAt:]

    endAt = startAt
    for p_i, p in enumerate(ramps[startAt:]):
        if p.endswith(visitNPat):
            logger.warning(f'matched end {p} against {visitNPat}')
            endAt += p_i + 1
            break

    return ramps[startAt:endAt]


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
