import glob
import logging
import os
import pathlib

import fitsio
import numpy as np

logger = logging.getLogger('hxramp')

rootDir = "/data/ramps"
calibDir = "/data/pfsx/calib"
sitePrefix = "PFJB"
nightPattern = '20[12][0-9]-[01][0-9]-[0-3][0-9]'

class HxRamp(object):
    nrows = 4096
    ncols = 4096

    def __init__(self, fitsOrPath):
        """Basic H4 ramp object, wrapping a PFS PFxB FITS file.

        Used to be generic H2 and H4, but given IRP on the H4s I'm giving up on that.

        Args
        ----
        fitsOrPath : `fitsio.FITS` or path-like.
            The path to open as a FITS file or an existing FITS object to use.
        """
        self.logger = logging.getLogger('hxramp')

        if not isinstance(fitsOrPath, fitsio.FITS):
            fitsOrPath = fitsio.FITS(fitsOrPath)
        self.fits = fitsOrPath
        self.phdu = self.header()

        self.calcBasics()

    def __str__(self):
        return (f"HxRamp(filename={self.fits._filename}, nReads={self.nreads}, "
                f"interleave={self.interleaveRatio})")

    def __del__(self):
        """ fitsio caches open files, so try to close when we know we want to."""
        if self.fits is not None:
            self.fits.close()
            self.fits = None

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        """ fitsio caches open files, so try to close when we know we want to."""
        if self.fits is not None:
            self.fits.close()
            self.fits = None
        return True

    def calcBasics(self):
        """Figure out some basic properties of the ramp, using the first read or header."""

        try:
            self.interleaveRatio = self.phdu['W_H4IRPN']
            self.interleaveOffset = self.phdu['W_H4IRPO']
        except KeyError:
            self.logger.warn('header does not have interleave keys, using data and guessing offset.')

            read0 = self.dataN(0)
            irp0 = self.irpN(0)

            if irp0 == 0:
                self.interleaveRatio = 0
            else:
                self.interleaveRatio = read0.shape[1] // irp0.shape[1]
            self.interleaveOffset = self.interleaveRatio

        try:
            self.nchan = self.phdu['W_H4NCHN']
        except KeyError:
            self.logger.warn('header does not have nchannels key, using 32.')
            self.nchan = 32

    @property
    def nreads(self):
        """Number of reads in ramp.

        This is WRONG: we are about to start reading and saving the reset frame.
        Also, does not handle "raw" frames, with the reference pixels leftt interleaved.

        """

        if self.interleaveRatio > 0:
            return (len(self.fits)-1)//2
        else:
            return (len(self.fits)-1)

    def _readIdxToAbsoluteIdx(self, n):
        """Convert possibly negative 0-indexed ramp read index into positive 0-indexed read"""
        nums = range(0, self.nreads)
        return nums[n]

    def _readIdxToFITSIdx(self, n):
        """Convert possibly negative 0-indexed ramp read index into positive 1-indexed HDU"""
        return self._readIdxToAbsoluteIdx(n) + 1

    def header(self, readNum=None):
        if readNum is None:
            return self.fits[0].read_header()
        else:
            idx = self._readIdxToFITSIdx(readNum)
            return self.fits[f'IMAGE_{idx}'].read_header()

    def dataN(self, n):
        """Return the data plane for the n-th read.

        Args
        ----
        n : `int`
          0-indexed read number

        Returns
        -------
        im : np.uint16 image
        """
        n = self._readIdxToFITSIdx(n)
        extname = f'IMAGE_{n}'
        return self.fits[extname].read()

    def irpN(self, n, raw=False, refPix4=False):
        """Return the reference plane for the n-th read.

        If the IRP HDU is empty we did not acquire using IRP. So return 0.

        Does not interpolate N:1 IRP planes to full size images.

        Args
        ----
        n : `int`
          0-indexed read number
        raw : `bool`
          If True, do not process/interpolate the raw IRP image.
        refPix4 : `bool`
          If True, return the `refpix4` image, based on the border pixels.
          Very unlikely to be what you want.

        Returns
        -------
        im : np.uint16 image, or 0 if there is none.

        """

        if refPix4:
            dataImage = self.dataN(n).astype('f4')
            corrImage, *_ = refPixel4(dataImage)
            return corrImage - dataImage  # Meh. Should have refPixel4 return the full correction image?

        n = self._readIdxToFITSIdx(n)
        extname = f'REF_{n}'
        try:
            irpImage = self.fits[extname].read()
        except:
            irpImage = None

        if raw:
            pass
        elif self.interleaveRatio != 1:
            irpImage0 = irpImage
            irpImage = constructFullIrp(irpImage, self.nchan,
                                        refPix=self.interleaveOffset)

        if irpImage is None or irpImage.shape == (1,1):
            return np.uint16(0)
        else:
            return irpImage

    def readN(self, n, doCorrect=True):
        """Return the IRP-corrected image for the n-th read.

        Note that data - ref is often negative, so we convert to float32 here.

        Args
        ----
        n : `int`
          0-indexed read number

        Returns
        -------
        im : np.float32 image
        """

        data = self.dataN(n).astype('f4')

        if self.interleaveRatio > 0:
            if doCorrect:
                data = data - self.irpN(n).astype('f4')
        else:
            if doCorrect:
                corrected, *_ = refPixel4(data)  # Beware: refPixel4 is pretty bad.
                data = corrected
        return data

    def cdsN(self, r0=0, r1=-1):
        """Return the CDS image between two reads.

        This is the most common way to get an quick image from an IRP H4 ramp, but is
        not the *right* way to do it.
        See .readStack() to get closer to that.

        Args
        ----
        r0 : `int`
          0-indexed read number of the 1st read
        r1 : `int`
          0-indexed read number of the 2st read


        Returns
        -------
        im : np.float32 image
        """
        return self.readN(r1) - self.readN(r0)

    def cds(self):
        """Return all the flux in the ramp."""
        return self.cdsN(r0=0, r1=-1)

    def dataStack(self, r0=0, r1=-1, dtype='u2'):
        """Return all the data frames, in a single 3d stack.

        Args
        ----
        r0 : `int`
          The 0-indexed read to start from.
        r1 : `int`
          The 0-indexed read to end with
        dtype : `str`
          If set and not "u2", the dtype to coerce to.

        Returns
        -------
        stack : the 3-d stack, with axis 0 being the reads.
        """

        r0 = self._readIdxToAbsoluteIdx(r0)
        r1 = self._readIdxToAbsoluteIdx(r1)
        nreads = r1 - r0 + 1

        stack = np.empty(shape=(nreads,self.ncols,self.nrows), dtype=dtype)
        for r_i in range(r0, r1+1):
            read = self.dataN(r_i)
            stack[r_i,:,:] = read

        return stack

    def irpStack(self, r0=0, r1=-1, dtype='u2', raw=False, refPix4=False):
        """Return all the reference frames, in a single 3d stack.

        Args
        ----
        r0 : `int`
          The 0-indexed read to start from.
        r1 : `int`
          The 0-indexed read to end with
        dtype : `str`
          If set and not "u2", the dtype to coerce to.
        raw : `bool`
          If True, do not interpolate/proceess the reference images
        refPix4 : `bool`
          If True, return the refPixel4 corrections.

        Returns
        -------
        stack : the 3-d stack, with axis 0 being the reads.
        """

        if refPix4 and dtype != 'f4':
            raise ValueError('irpRamps using refPixel4 cannot be unsigned shorts')
        r0 = self._readIdxToAbsoluteIdx(r0)
        r1 = self._readIdxToAbsoluteIdx(r1)
        nreads = r1 - r0 + 1

        stack = np.empty(shape=(nreads,self.ncols,self.nrows), dtype=dtype)
        for r_i in range(r0, r1+1):
            read = self.irpN(r_i, raw=raw, refPix4=refPix4)
            stack[r_i,:,:] = read

        return stack

    def readStack(self, r0=0, r1=-1):
        """Return all the ref-corrected frames, in a single 3d stack.

        Note that there will be one fewer reads than in the data: r0
        is subtracted from all later reads.

        This is probably close to where proper reductions will start
        from: all the reference-corrected reads in a single
        stack. Easy to pick up CRs, or to apply linearity,
        etc. corrections.

        If we were in space, could then simply fit lines through the
        pixels (or do something as trivial as
        np.mean(np.diff(axis=0)).

        Args
        ----
        r0 : `int`
          The 0-indexed read to start from.
        r1 : `int`
          The 0-indexed read to end with

        Returns
        -------
        stack : the 3-d stack, with axis 0 being the reads. Always 'f4'.

        """

        r0 = self._readIdxToAbsoluteIdx(r0)
        r1 = self._readIdxToAbsoluteIdx(r1)
        nreads = r1 - r0 + 1

        stack = np.empty(shape=(nreads,self.ncols,self.nrows), dtype='f4')
        for r_i in range(r0, r1+1):
            read1 = self.readN(r_i)
            stack[r_i-1,:,:] = read1

        return stack

    def cdsStack(self, r0=0, r1=-1):
        """Return all the CDS frames, in a single 3d stack.

        Note that there will be one fewer reads than in the data: r0
        is subtracted from all later reads.

        Args
        ----
        r0 : `int`
          The 0-indexed read to subtract from subsequent reads.
        r1 : `int`
          The 0-indexed read to end with

        Returns
        -------
        stack : the 3-d stack, with axis 0 being the reads. Always 'f4'.

        """

        r0 = self._readIdxToAbsoluteIdx(r0)
        r1 = self._readIdxToAbsoluteIdx(r1)
        nreads = r1 - r0

        stack = np.empty(shape=(nreads,self.ncols,self.nrows), dtype='f4')
        read0 = self.readN(r0)
        for r_i in range(r0+1, r1+1):
            read = self.readN(r_i)
            stack[r_i-1,:,:] = read - read0

        return stack

def interpolateChannelIrp(rawChan, refRatio, refOffset, doFlip=True):
    """Given a channel's IRP image and the IRP geometry, return a full-size reference image for the channel.

    Args
    ----
    rawChan : array
     The raw IRP channel, with the columns possibly subsampled by an integer factor.
    refRatio : int
     The factor by which the reference pixels are subsampled.
    refOffset : int
     The position of the reference pixel w.r.t. the associated science pixels.
    doFlip : bool
     Whether the temporal order of the columns is right-to-left.

    Returns
    -------
    im : the interpolated reference pixel channel.

    We do not yet know how to interpolate, so simply repeat the pixel refRatio times.

    """

    if refRatio == 1:
        return rawChan

    irpHeight, irpWidth = rawChan.shape
    refChan = np.empty(shape=(irpHeight, irpWidth * refRatio), dtype=rawChan.dtype)

    if doFlip:
        rawChan = rawChan[:, ::-1]

    for i in range(0, refRatio):
        refChan[:, i::refRatio] = rawChan

    if doFlip:
        refChan = refChan[:, ::-1]

    return refChan

def constructFullIrp(rawIrp, nChannel=32, refPix=None, oddEven=True):
    """Given an IRP image, return fullsize IRP image.

    Args
    ----
    rawImg : ndarray
      A raw read from the ASIC, possibly with interleaved reference pixels.
    nChannel : `int`
      The number of readout channels from the H4
    refPix : `int`
      The number of data pixels to read before a reference pixel.
    oddEven : `bool`
      Whether readout direction flips between pairs of amps.
      With the current Markus Loose firmware, that is always True.

    Returns
    -------
    img : full 4096x4096 image.

    - The detector was read out in nChannel channels, usually 32, but possibly 16, 4, or 1.

    - If oddEven is set (the default), the read order from the
      detector of pairs of channels is --> <--. The ASIC "corrects"
      that order so that the image always "looks" right: the
      column-order of the image is spatially, not temporally correct.

    - the N:1 ratio of science to reference pixels is deduced from the size of the image.

    - refPix tells us the position of the reference pixel within the N
      science pixels. It must be >= 1 (there must be at least one
      science pixel before the reference pixel). The ASIC default is
      for it to be the last pixel in the group.

    """

    logger = logging.getLogger('constructIRP')
    logger.setLevel(logging.DEBUG)

    h4Width = 4096
    height, width = rawIrp.shape

    # If we are a full frame, no interpolation is necessary.
    if width == h4Width:
        return rawIrp

    dataChanWidth = h4Width // nChannel
    refChanWidth = width // nChannel
    refRatio = h4Width // width
    refSkip = refRatio + 1

    if refPix is None:
        refPix = refRatio
    logger.debug(f"constructIRP {rawIrp.shape} {nChannel} {dataChanWidth} {refChanWidth} {refRatio} {refPix}")

    refChans = []
    for c_i in range(nChannel):
        rawChan = rawIrp[:, c_i*refChanWidth:(c_i+1)*refChanWidth]
        doFlip = oddEven and c_i%2 == 1

        # This is where we would intelligently interpolate.
        refChan = interpolateChannelIrp(rawChan, refRatio, refPix, doFlip)
        refChans.append(refChan)

    refImg = np.hstack(refChans)

    logger.debug(f"constructIRP {rawIrp.shape} {refImg.shape}")

    return refImg

def splitIRP(rawImg, nChannel=32, refPix=None, oddEven=True):
    """Given a single read from the DAQ, return the separated data and the reference images.

    Args
    ----
    rawImg : ndarray
      A raw read from the ASIC, possibly with interleaved reference pixels.
    nChannel : `int`
      The number of readout channels from the H4
    refPix : `int`
      The number of data pixels to read before a reference pixel.
    oddEven : `bool`
      Whether readout direction flips between pairs of amps.

    The image is assumed to be a full-width 4k read: IRP reads can only be full width.
    """

    logger = logging.getLogger('splitIRP')

    h4Width = 4096
    height, width = rawImg.shape

    # Can be no IRP pixels
    if width <= h4Width:
        return rawImg, None

    dataWidth = h4Width
    dataChanWidth = dataWidth // nChannel
    rawChanWidth = width // nChannel
    refChanWidth = rawChanWidth - dataChanWidth
    refRatio = dataChanWidth // refChanWidth
    refSkip = refRatio + 1

    if refPix is None:
        refPix = refRatio
    logger.info(f"splitIRP {rawImg.shape} {nChannel} {dataChanWidth} "
                f"{rawChanWidth} {refChanWidth} {refRatio} {refPix}")

    refChans = []
    dataChans = []
    for c_i in range(nChannel):
        rawChan = rawImg[:, c_i*rawChanWidth:(c_i+1)*rawChanWidth]
        doFlip = oddEven and c_i%2 == 1

        if doFlip:
            rawChan = rawChan[:, ::-1]
        refChan = rawChan[:, refPix::refSkip]

        dataChan = np.empty(shape=(height, dataChanWidth), dtype='u2')
        dataPix = 0
        for i in range(0, refRatio):
            # Do not copy over reference pixels, wherever they may be.
            if i == refPix:
                continue
            dataChan[:, dataPix::refRatio] = rawChan[:, i::refSkip]
            dataPix += 1

        if doFlip:
            refChan = refChan[:, ::-1]
            dataChan = dataChan[:, ::-1]

        refChans.append(refChan)
        dataChans.append(dataChan)

    refImg = np.hstack(refChans)
    dataImg = np.hstack(dataChans)

    logger.info(f"splitIRP {rawImg.shape} {dataImg.shape} {refImg.shape}")

    return dataImg, refImg

def ampSlices(im, ampN, nAmps=32):
    height, width = im.shape
    ampWidth = width//nAmps
    slices = slice(height), slice(ampN*ampWidth, (ampN+1)*ampWidth)

    return slices

def refPixel4(im, doRows=True, doCols=True, nCols=4, nRows=4, colWindow=4):
    """ Apply Teledyne's 'refPixel4' scheme.

    Step 1:
       For each amp, average all 8 top&bottom rows to one number.
       Subtract that from the amp.

    Step 2:
        Take a 9-row running average of the left&right rows.
        Subtract that from each row.

    This looks so very wrong: the 8 rows/columns are wildly different. But
    we are just here to duplicate their logic.
    """

    im = im.astype('f4')
    corrImage = np.zeros_like(im)
    imHeight, imWidth = im.shape

    rowRefs = np.zeros((nRows*2, imHeight), dtype='f4')
    rowRefs[0:nRows,:] = im[4-nRows:4,:]
    rowRefs[nRows:,:] = im[-nRows:,:]

    ampRefMeans = []
    for amp_i in range(32):
        slices = ampSlices(im, amp_i)
        ampImage = im[slices].copy()
        ampRefMean = (ampImage[4-nRows:4,:].mean()
                      + ampImage[-nRows:,:].mean()) / 2
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

def medianCubes(paths, r0=0, r1=-1):
    """ Given visits, return a supervisit of CDSs, where each CDS is the median of the visit CDSs.

    This is for building quick superdarks. It tries to be space-efficent, by handling each
    read independently.
    """

    ramps = [HxRamp(p) for p in paths]
    if 1 != len(set([r.nreads for r in ramps])):
        raise ValueError('all ramps must have the same number of reads')

    ramp0 = ramps[0]
    nreads = ramp0.nreads
    _reads = range(nreads)
    r0 = _reads[r0]
    r1 = _reads[r1]

    nvisits = len(paths)
    read0 = ramp0.readN(r0)

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

    def isr(self, visit, matchLevel=False, scaleLevel=False):
        path = rampPath(visit, cam=self.cam)
        _ramp = HxRamp(path)

        nreads = _ramp.nreads
        cds = _ramp.cds()

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
    r0corr, *_ = refPixel4(r0)
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

def badMask(ramp, r0thresh=5000):
    """Build up a mask from a set of tests, with each represented by a bitplane.

    Currently just:
     - r0mask = 0x01, where bright pixels in the 0th data read are marked bad.
    """

    _, r0mask = badMask_r0(ramp, thresh=r0thresh)

    finalMask = (r0mask * 0x01).astype('i4')

    return finalMask

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
