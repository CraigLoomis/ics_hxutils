from functools import reduce
import glob
import logging
import os
import pathlib

import astropy.io.fits as pyfits
import numpy as np
import fitsio

logger = logging.getLogger('hxstack')

rootDir = "/data/pfsx"
calibDir = "/data/pfsx/calib"
sitePrefix = "PFJB"
nightPattern = '20[12][0-9]-[01][0-9]-[0-3][0-9]'

class HxCalib(object):
    def __init__(self, cam=None, badMask=None, darkStack=None):
        self.cam = cam
        self.badMask = badMask
        self.darkStack = darkStack
        
    def isr(self, visit, matchLevel=False, scaleLevel=False):
        path = rampPath(visit, cam=self.cam)
        _ramp = ramp(path)
        
        nreads = rampNreads(_ramp)
        cds = rampCds(_ramp)
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

class xxxDarkCube(object):
    def __init__(self, cube, visits=None, center=None):
        self._cube = cube
        self.visits = visits
        self.nreads = len(self._cube)
        self.center = center
        
    def __getitem__(self, ii):
        return self.cds(r1=ii)
    
    def cds(self, r0=0, r1=-1):
        if r0 != 0:
            raise ValueError(f'can only take full CDS from DarkCubes, not with r0={r0}')
        return self._cube[r1]
    
def sliceIndices(seqOrLength, _slice):

    if not isinstance(_slice, slice):
        return _slice
    
    start = _slice.start
    stop = _slice.stop
    step = _slice.step

    if isinstance(seqOrLength, int):
        seqLen = seqOrLength
    elif isinstance(seqOrLength, (list, tuple)):
        seqLen = len(seqOrLength)

    if start is None:
        start = (0 if step > 0 else seqLen-1)
    elif start < 0:
        start += seqLen

    if stop is None:
        stop = (seqLen if step > 0 else -1)  # really -1, not last element
    elif stop < 0:
        stop += seqLen
    if stop > seqLen:
        stop = seqLen
        
    if step is None:
        step = 1
    return range(start, stop, step)
            
def sigClip(data, sigma=3.0, reps=3):
    keepMask = np.full_like(data, True, dtype=bool)

    for i in range(reps):
        mn = np.mean(data[keepMask])
        std = np.std(data[keepMask])

        keepMask[np.abs(data-mn) > sigma*std] = False

    return keepMask
        
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
    if visit == -1:
        night = lastNight()
        fileGlob = '[0-9][0-9][0-9][0-9][0-9][0-9]'
    else:
        night = nightPattern
        fileGlob = '%06d' % visit

    if cam is None:
        fileGlob = f'{fileGlob}[0-9][0-9]'
    else:
        armNums = dict(b=1, r=2, n=3, m=4)
        fileGlob = '%s%d%d' % (fileGlob, int(cam[1]), armNums[cam[0]])
                               
    ramps = glob.glob(os.path.join(rootDir, night, '%s%s.f*' % (prefix, fileGlob)))
    if visit == -1:
        return sorted(ramps)[-1]
    else:
        return ramps[0]

def lastRamp(prefix=None, cam=None):
    return rampPath(visit=-1, cam=cam, prefix=prefix)

def ramp(rampId, cam=None):
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
        pathOrFits = rampPath(rampId, cam=cam)
    else:
        pathOrFits = rampId
        
    if isinstance(pathOrFits, (str, pathlib.Path)):
        return fitsio.FITS(pathOrFits)
    else:
        return pathOrFits

def rampStep(pathOrFits, r0=0, r1=-1, call=None, 
             singleReads=None, doCorrect=True):
    """Call a function with every read in a ramp """
    r = ramp(pathOrFits)
    reads = range(len(r))
    r0 = reads[r0] 
    r1 = reads[r1]
    
    for r_i in range(r0, r1+1):
        if singleReads:
            read1 = rampRead(r, r_i, doCorrect=doCorrect)
        else:
            if r_i == r0:
                continue
            read1 = rampCds(r, r0, r_i, doCorrect=doCorrect)
            
        yield call(read1, r, r_i)
            
def rampHeaders(pathOrFits):
    r = ramp(pathOrFits)
    hlist = []
    for i in len(r):
        hlist.append(r[i].readHeader())

    return hlist
                     
def rampTimes(pathOrFits):
    hlist = rampHeaders(pathOrFits)
    times = []

    for h_i, h in enumerate(hlist):
        try:
            time1 = h['OBSTIME']
        except Exception as e:
            logger.WARN(f'failed to fetch time in HDU {h_i}: {e}')
            time1 = np.NAN
        times.append(time1)
        
    return np.array(times)
    
def readIdxToCount(ff, idx, r0=0):
    """ Return the right HDU and 0-indexed offset for a given index.

    Notes
    -----
    - Assert that the image HDUs in the file start at 0-indexed 1
    - 
    """
    assert not ff[0].has_data()

    hduList = np.arange(r0, len(ff)-1)
    return hduList[idx]

def readIdxToHduIdx(ff, idx, r0=0):
    return readIdxToCount(ff, idx, r0=r0)+1

def rampNreads(pathOrFits):
    ff = ramp(pathOrFits)
    return len(ff)-1

def rampRead(pathOrFits, readNumber=-1, doCorrect=True):
    ff = ramp(pathOrFits)

    fitsIdx = readIdxToHduIdx(ff, readNumber)
    read = ff[fitsIdx].read().astype('f4')
    height, width = read.shape

    if doCorrect is False:
        return read
    
    if width > height:
        return irpCorrect(read)
    else:
        return refPixel4(read)[0]

def rampCds(pathOrFits, r0=0, r1=-1, doCorrect=True):
    ff = ramp(pathOrFits)

    read0 = rampRead(ff, r0, doCorrect=doCorrect)
    read1 = rampRead(ff, r1, doCorrect=doCorrect)
    dread = read1-read0

    return dread

def rampCube(pathOrFits, r0=0, r1=-1, doCorrect=True):
    ff = ramp(pathOrFits)
    reads = range(rampNreads(ff))
    r0 = reads[r0]
    r1 = reads[r1]
    
    shape = rampRead(ff, r0).shape
    nreads = r1 - r0 + 1
    
    cube = np.zeros((nreads, shape[0], shape[1]), dtype='f4')
    for i in range(r0, r1+1):
        cube[i,:,:] = rampRead(ff, readNumber=i, doCorrect=doCorrect)
        
    return cube
    
def rampCdsCube(pathOrFits, r0=0, r1=-1, doCorrect=True):
    ff = ramp(pathOrFits)
    reads = range(rampNreads(ff))
    r0 = reads[r0]
    r1 = reads[r1]
    
    read0 = rampRead(ff, r0, doCorrect=doCorrect)
    shape = read0.shape
    nreads = r1 - r0 + 1
    
    cube = np.zeros((nreads-1, shape[0], shape[1]), dtype='f4')
    for i_i, i in enumerate(range(r0+1, r1+1)):
        read1 = rampRead(ff, readNumber=i, doCorrect=doCorrect)
        cube[i_i,:,:] = read1 - read0
        
    return cube
    
def medianCubes(visits, r0=0, r1=-1, cam=None, doCorrect=True):
    """ Given visits, return a supervisit of CDSs, where each CDS is the median of the visit CDSs. 

    This is for building superdarks. It tries to be space-efficent, by handling each 
    read independently.
    """

    ramps = [ramp(rampPath(v, cam=cam)) for v in visits]
    if 1 != len(set([rampNreads(r) for r in ramps])):
        raise ValueError('all ramps must have the same number of reads')
    
    ramp0 = ramps[0]
    _reads = range(rampNreads(ramp0))
    r0 = _reads[r0]
    r1 = _reads[r1]

    nvisits = len(visits)
    nreads = len(ramp0)
    read0 = rampRead(ramps[0], r0)
    tempStack = np.empty(shape=(nvisits, *(read0.shape)), 
                         dtype=read0.dtype)
    del read0
    for read_i in range(r0, r1+1):
        if read_i == r0:
            read0s = [rampRead(ramp, r0, doCorrect=doCorrect) for ramp in ramps]
            outStack = np.empty(shape=(nreads-1, *(read0s[0].shape)), dtype=read0s[0].dtype)
            continue
        for ramp_i, ramp1 in enumerate(ramps):
            read1 = rampRead(ramp1, read_i, doCorrect=doCorrect)
            cds1 = read1-read0s[ramp_i]
            if ramp_i == 0:
                cdsRefMed = np.median(cds1)
            else:
                dmed = np.median(cds1) - cdsRefMed
                print(f"dmed({read_i}:{ramp_i}) = {dmed:0.3f}")
                cds1 -= dmed
            tempStack[ramp_i, :, :] = cds1
            
        outStack[read_i, :, :] = np.median(tempStack, axis=0)
    
    return DarkCube(outStack)
        
    # for v_i, v in visits
                
def irpCorrect(im):
    height, width = im.shape

    if width > height:
        data, ref = refSplit(im)
        return data - ref
    else:
        return im
    
def refSplit(im, evenOdd=True):
    height, width = im.shape
    nAmps = 32                  # All that is implemented in firmware
    if width != 2*height:
        raise RuntimeError('can only deal with 1-1 IRP')
    
    if evenOdd:
        rawAmpWidth = width // nAmps
        ampWidth = rawAmpWidth // 2
        
        refIm = np.zeros(shape=(im.shape[0], im.shape[1]//2), dtype=im.dtype)
        actIm = np.zeros(shape=(im.shape[0], im.shape[1]//2), dtype=im.dtype)

        for a_i in range(nAmps):
            oneAmp = im[:,a_i*rawAmpWidth:(a_i+1)*rawAmpWidth]
        
            if a_i % 2 == 1:
                refXs = slice(0,None,2)
                actXs = slice(1,None,2)
            else:
                actXs = slice(0,None,2)
                refXs = slice(1,None,2)

            # print(f"amp {a_i} {oneAmp.shape} {a_i*rawAmpWidth} {(a_i+1)*rawAmpWidth} {actXs} {refXs}")
            refIm[:,a_i*ampWidth:(a_i+1)*ampWidth] = oneAmp[:,refXs]
            actIm[:,a_i*ampWidth:(a_i+1)*ampWidth] = oneAmp[:,actXs]
    else:
        refIm = im[:,::2]
        actIm = im[:,1::2]
        
    return actIm, refIm

def ampSlices(im, ampN, nAmps=32):
    height, width = im.shape
    ampWidth = width//nAmps
    slices = slice(height), slice(ampN*ampWidth, (ampN+1)*ampWidth+1)

    return slices

def refPixel4(im, doRows=True, doCols=True, nCols=4, nRows=4, colWindow=4):
    """ Apply Teledyne's 'refPixel4' scheme. 
    
    Step 1:
       For each amp, average all 8 top&bottom rows to one number. 
       Subtract that from the amp.
       
    Step 2:
        Take a 9-row running average of the left&right rows.
        Subtract that from each row.
    """
    corrImage = im * 0
    imHeight, imWidth = im.shape
    
    rowRefs = np.zeros((nRows*2, imHeight), dtype='f4')
    rowRefs[0:nRows,:] = im[4-nRows:4,:]
    rowRefs[nRows:,:] = im[-nRows:,:]

    ampRefMeans = []
    for amp_i in range(32):
        slices = ampSlices(im, amp_i)
        ampImage = im[slices].copy()
        ampRefMean = (ampImage[4-nRows:4,:].mean() + 
                      ampImage[-nRows:,:].mean()) / 2
        ampRefMeans.append(ampRefMean)
        ampImage -= ampRefMean
        corrImage[slices] = ampImage
    corr1Image = corrImage - im

    if not doRows:
        corrImage = im.copy()
        
    sideRefImage = np.ndarray((imHeight, nCols*2), dtype=im.dtype)
    sideRefImage[:, :nCols] = corrImage[:, 4-nCols:4]
    sideRefImage[:, -nCols:] = corrImage[:, -nCols:]
    sideCorr = np.zeros((imHeight,1))
    for row_i in range(colWindow, imHeight-colWindow+1):
        sideCorr[row_i] = sideRefImage[row_i-colWindow:row_i+colWindow,:].mean()

        if doCols:
            corrImage[row_i, :] -= sideCorr[row_i]
    
    return corrImage, ampRefMeans, corr1Image, rowRefs, sideRefImage, sideCorr

class HxStack(object):
    def __init__(self, filename):
        self.filename = filename
        self._hdulist = None
        self._cube = None
        
        self.namps = 32
        self.width = self.ncols
        self.height = self.nrows
        
        self.fits = 'fitsio'
        self.readFile()
        
    def __str__(self):
        return "<%s(%s) = %dx%d,%d>" % (self.__class__.__name__,
                                        self.filename,
                                        self.width, self.height, self.namps)

    @property
    def cube(self):
        if self._cube is None:
            self.readCube()
            
        return self._cube

    @property
    def nreads(self):
        return len(self._hdulist)-1

    def normIdx(self, idx):
        return idx if idx >= 0 else self.nreads+idx
    
    def _parseGeom(self, hdu=None):
        return
    
        if hdu is None:
            hdu = self._hdulist[-1]
        if self.fits == 'pyfits':
            hdr = hdu.header
        else:
            hdr = hdu.read_header()
        self.hdr = hdr
        
        self.width = hdr['NAXIS1']
        self.height = hdr['NAXIS2']
        
        self.ampWidth = self.width // self.namps
    
    def readFile(self):
        if self.fits == 'pyfits':
            self._hdulist = pyfits.open(self.filename)
        else:
            self._hdulist = fitsio.FITS(self.filename)
        self._parseGeom()

    def _readPartialCube(self, readList=None, xslice=None, yslice=None):
        if readList is None:
            readList = range(self.nreads)

        height, width = self._hdulist[1].read().shape
        if xslice is None:
            xslice = slice(None, None)
        else:
            width = len(tuple(sliceIndices(width, xslice)))
            
        if yslice is None:
            yslice = slice(None, None)
        else:
            height = len(tuple(sliceIndices(height, yslice)))

        depth = len(readList)
        if depth * width * height == 0:
            raise RuntimeError("empty cube: %s %s %s %s" % (depth, height, width, readList))
        cube = np.zeros((depth, height, width), dtype='f4')

        for i,r_i in enumerate(readList):
            im = self._hdulist[r_i+1].read()
            cube[i,:,:] = im[yslice, xslice]

        return cube
    
    def readCube(self):
        self._cube = None
        self._cube = self._readPartialCube()
        return self._cube

    def getRead(self, readnum, dtype='f4', flipOddAmps=False):
        readnum = readIdxToHduIdx(readnum)

        if self.fits == 'pyfits':
            read = self._hdulist[readnum].data.astype(dtype)
        else:
            read = self._hdulist[readnum].read().astype(dtype)
        if flipOddAmps:
            for a in range(32):
                if a%2 == 1:
                    _, xs = self._ampSlices(a)
                    _, xsr = self._ampSlices(a, doReverse=False)
                    read[:, xs] = read[:, xsr]
                
        return read

    def cds(self, r0=0, r1=-1, dtype='f4'):
        return self.getRead(r1, dtype=dtype) - self.getRead(r0, dtype=dtype)
    
    def _ampSlices(self, ampNum, doReverse=True):
        yslice = slice(0,self.height)

        if doReverse:
            readDir = 1 if (ampNum%2 == 0) else -1
        else:
            readDir = 1
        xslice = np.arange(self.ampWidth*ampNum, self.ampWidth*(ampNum+1))
        if readDir == -1:
            xslice = xslice[::-1]
        
        return yslice, xslice

    def ampImage(self, ampNum, read0=0, read1=-1):
        ys, xs = self._ampSlices(ampNum)

        ampCube = self._readPartialCube(sliceIndices(self.nreads,
                                                     slice(read0, read1)),
                                        xslice=xs, yslice=ys)
        return ampCube
    
    def rejectCRs(self, rejectSigma=5.0):
        # 1. make the per-pixel diff ramp.
        # 2. get the per-pixel median diff and stddev.
        # 3. identify the diff outliers (the CR hits)
        # 4. replace CRs with median diffs (careful about 1st reads)
        # 5. reconstruct cube with cleaned diffs
        dcube = np.diff(self.cube, axis=0)
        dmedians = np.median(dcube, axis=0)
        dsigs = np.std(dcube, axis=0)
        
        dbad = dcube > rejectSigma*dsigs
        dbadIdx = np.where(dbad)
        badRead0 = dcube[0,:,:] < 0
        fixedDcube = dcube.copy()
        fixedDcube[dbadIdx] = np.broadcast_to(dmedians, dcube.shape)[dbadIdx]
    
        # TODO: deal with consecutive bad pixels at the start of a ramp, etc.
        newCube = np.zeros_like(self.cube)
        
        # Grr, some indexing gotcha so am being wasteful
        read0 = self.cube[0].copy()
        read1 = self.cube[1].copy()
        read0[badRead0] = read1[badRead0] - dmedians[badRead0]
        newCube[0,:,:] = read0
        
        # There is probably a clever way to do this...
        for i in range(self.cube.shape[0] - 1):
            newCube[i+1,:,:] = newCube[i] + dcube[i]

        # return the cleaned cube and the mask of replaced pixels.
        return newCube, dbad, dmedians, dcube

class H2Stack(HxStack):
    namps = 32
    ncols = 2048
    nrows = 2048
    ampWidth = nrows // namps

class H4Stack(HxStack):
    namps = 32
    ncols = 4096
    nrows = 4096
    ampWidth = nrows // namps

singleReadTime = 5.57
detectorGains = dict(n1=7.78, n9=7.78, n8=6.59) # uV/e-

def calcGain(preampGain, cam='n9'):
    detGain = detectorGains[cam]
    
    adcInput = detGain * preampGain # uV/e-
    adcRange = 2000000 # uV
    adcGain = adcInput / adcRange * 65536 # ADU/e-
    
    return 1/adcGain

def bithist(arr):
    nbits = 16
    pops = np.zeros((nbits,), dtype='i4')
    
    flatArr = arr.flatten()
    for i in range(nbits):
        pops[i] = np.sum(np.bitwise_and(flatArr,(1<<i)) > 0)
        
    return pops / len(flatArr)


def cdsNoise(ramp, r0=1, r1=2, doCorrect=False, rad=100, nSamples=1000):
    read0 = rampRead(ramp, r0, doCorrect=doCorrect)
    read1 = rampRead(ramp, r1, doCorrect=doCorrect)

    height, width = read0.shape
    if width > height:
        read0, _ = refSplit(read0)
        read1, _ = refSplit(read1)

    sigs = []
    mns = []
    stds = []
    dregs = []
    for i in range(nSamples):
        x = np.random.randint(rad+5, width-rad-5)
        y = np.random.randint(rad+5, height-rad-5)
        reg0 = read0[y-rad:y+rad+1, x-rad:x+rad+1]
        reg1 = read1[y-rad:y+rad+1, x-rad:x+rad+1]
        dreg = reg1 - reg0
        
        mns.append(np.median(dreg))
        stds.append(np.std(dreg))
        dregs.append(dreg)
        
    sigs = np.array(sigs)
    mns = np.array(mns)
    stds = np.array(stds)
    
    return np.median(mns), np.median(stds), mns, stds, dregs, sigs
