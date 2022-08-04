import logging

import numpy as np

from ics.hxutils import hxramp
from ics.hxutils import pathUtils

logger = logging.getLogger('hxcalib')

def medianCubes(paths, r0=0, r1=-1):
    """ Given visits, return a supervisit of CDSs, where each CDS is the median of the visit CDSs.

    This is for building quick superdarks. It tries to be space-efficent, by handling each
    read independently.
    """

    ramps = [hxramp.HxRamp(p) for p in paths]
    if 1 != len(set([r.nreads for r in ramps])):
        raise ValueError('all ramps must have the same number of reads')

    ramp0 = ramps[0]
    nreads = ramp0.nreads
    _reads = range(nreads)
    r0 = _reads[r0]
    r1 = _reads[r1]

    nvisits = len(paths)
    read0 = ramp0.readN(r0)

    print(f'dark cube: {nvisits} visits, {nreads} reads from {r0} to {r1}')
    # The stack of the frames fomr all the visits for a _single_ given read.
    tempStack = np.empty(shape=(nvisits, *(read0.shape)),
                         dtype=read0.dtype)
    del read0
    for read_i in range(r0, r1+1):
        if read_i == r0:
            read0s = [r.readN(r0) for r in ramps]
            outStack = np.empty(shape=(nreads, *(read0s[0].shape)), dtype=read0s[0].dtype)

            continue
        for ramp_i, ramp1 in enumerate(ramps):
            read1 = ramp1.readN(read_i)
            cds1 = read1-read0s[ramp_i]

            # Remove DC offsets between all instances of a given read.
            if ramp_i == 0:
                cdsRefMed = np.median(cds1)
            else:
                dmed = np.median(cds1) - cdsRefMed
                print(f"dmed({read_i}:{ramp_i}) = {dmed:0.3f}")
                cds1 -= dmed
            tempStack[ramp_i, :, :] = cds1

        outStack[read_i, :, :] = np.median(tempStack, axis=0)

    return outStack

class HxCalib(object):
    """Crudest possible calibration object """
    def __init__(self, cam=None, badMask=None, darkStack=None):
        self.cam = cam
        self.badMask = badMask
        self.darkStack = darkStack

    def isr(self, visit, matchLevel=False, scaleLevel=False, r0=0, r1=-1):
        path = pathUtils.rampPath(visit, cam=self.cam)
        _ramp = hxramp.HxRamp(path)

        nreads = _ramp.nreads
        cds = _ramp.cdsN(r0=r0, r1=r1)

        if self.darkStack is None:
            return cds

        dark = self.darkStack[nreads-1]

        if matchLevel or scaleLevel:
            cdsMed = np.median(cds)
            darkMed = np.median(dark)
            if scaleLevel:
                return cds - dark*(cdsMed / darkMed)
            elif matchLevel:
                return cds - (dark + (cdsMed - darkMed))
        else:
            return cds - dark

def badMask_r0(ramp, thresh=8000):
    """Use data0 to find some hot pixels: there should be no signal in that read."""

    r0 = ramp.dataN(0).astype('f4')

    # correct levels
    r0corr, *_ = hxramp.refPixel4(r0)
    r0corr -= np.median(r0corr)

    r0mask = (r0corr > thresh).astype('i4')

    return r0corr, r0mask

def badMask_noflux(ramp):
    """Look for pixels which do not accumulate flux or which invert.

    Hmm, needs the right ramp, and will overlap with the brightest data0 pixels.
    """

    data0 = ramp.dataN(0).astype('f4')
    data1 = ramp.dataN(-1).astype('f4')
    flux = data1-data0

    fluxMask = (flux <= 0).astype('i1')

    return fluxMask

def badMask_sigma(ramp, sigma=2):
    cds = ramp.cds()
    mask = np.zeros(shape=cds.shape, dtype='i1')

    patchSize = 128

    for y in range(0, 4096, patchSize):
        for x in range(0, 4096, patchSize):
            p1 = cds[y:y+patchSize, x:x+patchSize]
            pmed = np.median(p1)
            pdev = np.std(p1)
            clip = np.abs(p1 - pmed) >= pdev*sigma

            mask[y:y+patchSize, x:x+patchSize] = clip

    logger.info(f'{sigma} {mask.sum()}')
    return mask

def badMask(ramp, r0thresh=5000, sigma=3):
    """Build up a mask from a set of tests, with each represented by a bitplane.

    Currently just:
     - r0mask = 0x01, where bright pixels in the 0th data read are marked bad.
    """

    _, r0mask = badMask_r0(ramp, thresh=r0thresh)
    sigmask = badMask_sigma(ramp, sigma=sigma)
    # lowmask = badMask_noflux(ramp)
    finalMask = (r0mask * 0x01 | sigmask * 0x02).astype('i4')

    return finalMask

singleReadTime = 5.429
detectorGains = dict(n1=7.78, n3=7.78, n9=7.78, n8=6.59) # uV/e-

def calcGain(preampGain, cam):
    logger.warning(f'BAD Craig is still using cam, not detector id.')
    detGain = detectorGains[cam]

    adcInput = detGain * preampGain # uV/e-
    adcRange = 2200000 # uV
    adcGain = adcInput / adcRange * 65536 # ADU/e-

    return 1/adcGain
