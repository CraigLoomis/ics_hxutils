from importlib import reload
import logging
import os
import pathlib

import fitsio
import numpy as np
import scipy
import matplotlib.pyplot as plt

import sep

from . import pathUtils
from . import nirander
from . import hxcalib
from . import hxramp

reload(hxcalib)
reload(hxramp)
reload(nirander)

logger = logging.getLogger('hxdisplay')
logger.setLevel(logging.INFO)

def regStats(fpath, slices=None, r0=0, r1=-1, fitr0=1, fitr1=-1, doDiffs=False,
             residMax=-1, order=1, preampGain=True):

    ramp = hxramp.HxRamp(fpath)
    nreads = ramp.nreads
    width = ramp.width

    readTime = ramp.frameTime
    if preampGain is False:
        preampGain = 1.0
        adcGain = 1.0
    elif preampGain is not True:
        adcGain = hxcalib.calcGain(preampGain, ramp.cam)
    else:
        preampGain = ramp.preampGain
        adcGain = hxcalib.calcGain(preampGain, ramp.cam)

    print(f"hreads={nreads} readTime={readTime} adcGain={adcGain}")
    fig, plots = plt.subplots(nrows=2, figsize=(10,6), sharex=True)
    p1, p2  = plots

    if r0 > fitr0:
        fitr0 = r0
    else:
        fitr0 -= r0

    if slices is None:
        slices = slice(4,4093), slice(4,4093)
    meds = []
    dmeds = []
    ddevs = []
    if r1 == -1 or r1 is None:
        plotReads = np.arange(r0+1, nreads)
    else:
        plotReads = np.arange(r0+1, r1+1)
    print(plotReads)
    read0 = ramp.readN(r0)[slices]
    meds.append(np.median(read0))
    for r_i in plotReads:
        read1 = ramp.readN(r_i)[slices]
        meds.append(np.median(read1))
        dim = read1 - read0
        dmeds.append(np.median(dim))
        ddevs.append(np.std(dim))
    dmeds = np.array(dmeds)
    ddevs = np.array(ddevs)
    yoffset = dmeds[fitr0]

    plotX = plotReads * readTime
    fitY = np.array(dmeds[fitr0:fitr1])
    fitX = plotX[fitr0:fitr1]
    coeffs = np.polyfit(fitX, fitY, order)
    print(f'x: {fitX}')
    print(f'y: {fitY}')

    label = "%0.3ft + %0.2f" % (coeffs[0], coeffs[1])
    line = np.poly1d(coeffs)
    # print(len(meds), len(fitX), len(fitY))

    p1.plot(plotX, dmeds - coeffs[-1], '+-')
    p1.plot(plotX, line(plotX) - coeffs[-1], ':', color='red', label=label)
    p1.plot(fitX, line(fitX) - coeffs[-1], color='red')

    residX = plotX
    residY = dmeds
    if residMax is not None:
        residX = plotX[:residMax]
        residY = dmeds[:residMax]
    p2.plot(residX, residY - line(residX), label='residuals')
    p2.hlines(0,fitX[0],fitX[-1],colors='r',alpha=0.5)
    title = "%s\nx=[%d,%d] y=[%d,%d]  nreads=%d %0.2f $e^-/ADU$ preampGain=%0.2f readTime=%0.2f" % (os.path.basename(fpath),
                                                                                                    slices[1].start,
                                                                                                    slices[1].stop-1,
                                                                                                    slices[0].start,
                                                                                                    slices[0].stop-1,
                                                                                                    nreads, adcGain,
                                                                                                    preampGain, readTime)
    p1.set_title(title)
    p1.grid(which='major', axis='both', alpha=0.5)
    p1.grid(which='minor', axis='y', alpha=0.25)
    p2.grid(which='major', axis='both', alpha=0.5)
    p2.grid(which='minor', axis='y', alpha=0.25)
    p1.legend(loc='lower right', fontsize='small')
    p2.legend(loc='lower right', fontsize='small')

    p1.set_ylabel(f'$e^- increment$')
    #p3.set_ylabel('$e^-/s$')
    p2.set_xlabel('seconds')

    # print(len(plotX), len(plotX[1:]), len(ddevs))
    #if doDiffs:
    #    p3.plot(plotX[1:], dmeds / ourReadTime)
    #else:
    #    p3.plot(plotX[1:], ddevs / ourReadTime)
    #p3.hlines(0,0,plotX[-1],colors='r',alpha=0.3)

    fig.tight_layout()

    return fig, fitX, fitY, coeffs

def regShow(stack, slices, display, r0=0, diffs='last', matchScales=True):
    display.set('frame delete all')
    display.set('tile grid; tile grid gap 1; tile yes')
    display.set('scale zscale')
    display.set('frame lock image')


    display.set('frame new')
    _read = stack.getRead(-1)
    read = _read[slices]
    display.set_np2arr(read)

    for i, r_i in enumerate(range(r0, stack.nreads)):
        display.set('frame new')
        _read = stack.getRead(r_i)
        read = _read[slices]
        if i == 0:
            med0 = np.median(read)
            read0 = read
            lastRead = read
        elif diffs == 'first':
            read -= read0
        elif diffs == 'prev':
            dread = read - lastRead
            lastRead = read
            read = dread
        else:
            read -= med0

        display.set_np2arr(read)

    if matchScales:
        display.set('match scalelimits')
        display.set('frame 2')
        display.set('scale zscale')
        display.set('frame 1')
        display.set('scale zscale')

def imStats(stack, r0=1, r1=-1,
            slices=None, width=None, doDark=True,
            sigClip=None):
    im1 = stack.getRead(r1)

    if r0 is None:
        dim = im1
        if doDark:
            dim -= darkCache[0]
    else:
        im0 = stack.getRead(r0)
        dim = im1-im0
        if doDark:
            dim -= darkCache[stack.normIdx(r1) - stack.normIdx(r0)]
    if slices:
        dim = dim[slices]

    fig, pl = plt.subplots()
    p0 = pl
    mn = np.mean(dim)
    sd = np.std(dim)

    if sigClip is not None:
        flatIm = dim.flatten()
        keep_w = np.where((flatIm > mn-sigClip*sd) & (flatIm < mn+sigClip*sd))
        mn = np.mean(flatIm[keep_w])
        sd = np.std(flatIm[keep_w])

        print("clipped %d -> %d" % (len(flatIm), len(keep_w[0])))

    if width is None:
        xrange = mn-3*sd, mn+3*sd
    elif width is False:
        xrange = None
    else:
        xrange = mn-width, mn+width

    print("%0.3f %0.3f" % (mn, sd))
    p0.hist(dim.flatten(), normed=True,
            bins=100, range=xrange,
            label="$\mu=%0.2f$ $\sigma=%0.2f$ $kept=%0.2f$" % (mn, sd,
                                                               (1 if sigClip is None else len(keep_w[0])/len(flatIm))))

    p0.legend()

    return fig, mn, sd

def isrShow(fpath, display, hxCalib, removeBackground=False, r0=0, r1=-1):
    im = hxCalib.isr(fpath)
    if removeBackground:
        bkgnd = sep.Background(im, mask=hxCalib.badMask)
        im =  im - bkgnd

    display.set('frame delete all')
    display.set('frame new')
    display.set('scale zscale')
    display.set_np2arr(im)

def imShow(fpath, display, r0=1, r1=-1, darkRamp=None,
           showAll=False, doCorrect=True, matchDarkLevel=False):
    stack = hx.ramp(fpath)

    display.set('frame delete all')
    display.set('tile grid; tile yes')
    display.set('zoom to fit')
    display.set('frame lock image')

    if showAll:
        display.set('frame new')
        display.set('scale zscale')
        display.set_np2arr(hx.rampRead(stack, r0, doCorrect=doCorrect))

        display.set('frame new')
        display.set('scale zscale')
        display.set_np2arr(hx.rampRead(stack, r1, doCorrect=doCorrect))

        display.set('frame new')
        display.set('scale zscale')
        cds1 = hx.rampCds(stack, r0=r0, r1=r0+1, doCorrect=doCorrect)
        display.set_np2arr(cds1)

    display.set('frame new')
    display.set('scale zscale')
    cds = hx.rampCds(stack, r0=r0, r1=r1, doCorrect=doCorrect)
    display.set_np2arr(cds)

    if darkRamp is not None:
        display.set('frame new')
        darkSum = darkRamp.cds(r0=r0, r1=r1)
        if matchDarkLevel:
            scale = np.median(cds)/np.median(darkSum)
            print(f"adjusting dark level by {scale:0.6f} ")
            darkSum *= scale
        display.set_np2arr(cds-darkSum)

        if showAll:
            display.set('frame new')
            display.set_np2arr(darkSum)

def plotRefs(fname, reads=None, refRows=None, r0=0):
    fig, plots = plt.subplots(nrows=4, sharex=True,
                              figsize=(10,8))
    p1, p2, p3, p4 = plots

    refWidth = 4

    stack = hx.ramp(fname)

    if refRows is None:
        refRows = range(refWidth)
    refRows = np.array(refRows, dtype='i2')

    if reads is None:
        reads = range(r0, hx.rampNreads(stack))
    for i in reads:
        im = hx.rampRead(stack, i)

        bottom = im[0:refWidth, :]
        top = im[-1:-refWidth-1:-1, :]

        left = im[:, 0:refWidth].T
        right = im[:, -1:-refWidth-1:-1].T

        #p1.set_prop_cycle(None)
        for i in refRows:
            p1.plot(bottom[i])

        #p2.set_prop_cycle(None)
        for i in refRows:
            p2.plot(top[i])

        #p3.set_prop_cycle(None)
        for i in refRows:
            p3.plot(left[i])

        #p4.set_prop_cycle(None)
        for i in refRows:
            p4.plot(right[i])

    #p1.plot(np.median(bottom, axis=0), 'magenta', alpha=0.5)
    #p2.plot(np.median(top, axis=0), 'magenta', alpha=0.5)

    _yrange1 = p1.get_ylim()
    _yrange2 = p2.get_ylim()
    yrange = (min(_yrange1[0], _yrange2[0]),
              max(_yrange1[1], _yrange2[1]))
    p1.set_ylim(yrange)
    p2.set_ylim(yrange)

    _yrange1 = p3.get_ylim()
    _yrange2 = p4.get_ylim()
    yrange = (min(_yrange1[0], _yrange2[0]),
              max(_yrange1[1], _yrange2[1]))
    p3.set_ylim(yrange)
    p4.set_ylim(yrange)

    filename = os.path.basename(fname)
    p1.set_title('file=%s rows=%s' % (filename, refRows))

    fig.tight_layout()

        #p2.plot(np.median(left, axis=0))
        #p2.plot(np.median(right, axis=0))

def focusPlot(sweep, p=None, title=None, sizeOnly=True, dithers=False, yrange=None):
    if p is None:
        f,p1 = plt.subplots(num='focusPlot', clear=True)
    else:
        f = None
        p1 = p

    if len(sweep) == 0:
        return f, 0, None

    if title is None:
        title = f"{sweep['wavelength'].values[0]:0.0f} @ {sweep['row'].values[0]:0.0f}"

    sweep = sweep.sort_values(by=['focus'])

    minx, poly = nirander.getBestFocus(sweep)
    if not sizeOnly:
        p1.plot(sweep.focus, sweep.x2, '+-', alpha=0.3, label='x')
        p1.plot(sweep.focus, sweep.y2, '+-', alpha=0.3, label='y')
    p1.plot(sweep.focus, sweep['size'], '+-', alpha=0.75, label='size')

    xx = np.linspace(sweep.focus.min(), sweep.focus.max()+1, 100)
    yy = poly(xx)
    # print(f'{title}: {minx:0.2f}, {poly(minx):0.2f}')
    p1.plot(xx, yy, label=f'best={minx:0.0f}')
    p1.plot([minx], [poly(minx)], 'kx', markersize=10)
    p1.set_xlim(sweep.focus.min(), sweep.focus.max())
    p1.legend(fontsize='x-small')
    p1.grid(alpha=0.2)
    p1.set_title(title)
    if yrange is not None:
        p1.set_ylim(*yrange)
    elif dithers:
        p1.set_ylim(15, 35)
    else:
        p1.set_ylim(0, 5)

    return f, minx, poly(minx)

def focusOffsetPlot(meade, focusResults, visit=None):
    f, pl = plt.subplots(num='focusOffsetGrid', clear=True)
    x, y = meade.stepsToPix(focusResults['xstep'], focusResults['ystep'])
    val = pl.scatter(x.values, y.values, c=focusResults['focus'].values, cmap=plt.cm.get_cmap('viridis'))
    f.colorbar(val)
    f.suptitle(f'best focus, from visit={visit}')
    return f

def dispVisits(disp, visits, r0=0, r1=-1, cam='n1', 
               doClear=True, medFilter=None):
    if doClear:
        disp.set('frame delete all')
    disp.set('frame lock image')
    disp.set('tile grid')
    disp.set('tile yes')

    if isinstance(visits, int):
        visits = [visits]
    for v in visits:
        path = pathUtils.rampPath(v, cam=cam)
        cds = hxramp.HxRamp(path).cdsN(r0=r0, r1=r1)
        if medFilter is not None and medFilter > 0:
            cds = scipy.ndimage.median_filter(cds, medFilter)
        disp.set('frame new')
        disp.set_np2arr(cds)

def dispRamp(ramp, disp, reads=None):
    disp.set('frame delete all')
    disp.set('frame lock image')
    disp.set('tile grid')
    disp.set('tile yes')

    if reads is None:
        reads = range(1, ramp.nreads)
    for r in reads:
        cds = ramp.cds(r1=r)
        print(f'read r1={r} of {reads}: {cds.shape} {cds.dtype}')
        disp.set('frame new')
        disp.set_np2arr(cds)

def dispCds(ramp, disp, doClear=True, doCorrect=True, r0=0, r1=-1):
    d0 = ramp.readN(r0, doCorrect=doCorrect)
    d1 = ramp.readN(r1, doCorrect=doCorrect)

    if doClear:
        disp.set('frame delete all')
        disp.set('frame lock image')

    disp.set('frame new')
    disp.set_np2arr(d1-d0)

def dispIrpPanel(ramp, disp, r0=1, r1=-1):
    """Display the components of a CDS.

    Top row: the two Data images, and their diff.
    Middle row: the two IRP images and their diff
    Bottom row: the data-irp images for all three columns.

    The LR image is the net IRP-corrected CDS image.
    """

    disp.set('frame delete all')
    disp.set('frame lock image')
    disp.set('tile grid')
    disp.set('tile grid mode automatic')
    disp.set('tile yes')
    disp.set('scale zscale')
    disp.set('scale linear')

    d0 = ramp.dataN(r0)
    d1 = ramp.dataN(r1)

    i0 = ramp.irpN(r0)
    i1 = ramp.irpN(r1)

    r0 = ramp.readN(r0)
    r1 = ramp.readN(r1)


    disp.set('frame new')
    disp.set_np2arr(d0)
    disp.set('frame new')
    disp.set_np2arr(d1)
    disp.set('frame new')
    dd = d1.astype('f4')-d0
    disp.set_np2arr(dd)

    disp.set('frame new')
    disp.set_np2arr(i0)
    disp.set('frame new')
    disp.set_np2arr(i1)
    disp.set('frame new')
    di = i1.astype('f4')-i0
    disp.set_np2arr(di)

    disp.set('frame new')
    disp.set_np2arr(r0)
    disp.set('frame new')
    disp.set_np2arr(r1)
    disp.set('frame new')
    dr = r1-r0
    disp.set_np2arr(dr)

    return dd, di, dr

def dispDiff(ramp, disp, r0=0, r1=1, r2=2):
    d0 = ramp.readN(r0)
    d1 = ramp.readN(r1)
    d2 = ramp.readN(r2)

    dispPairs(disp, d0, d1, d2)

def dispCdsRamp(ramp, disp, r0=1, r1=-1):
    nums = range(ramp.nreads)
    r0 = nums[r0]
    r1 = nums[r1]

    disp.set('frame delete all')
    disp.set('frame lock image')
    disp.set('tile grid')
    disp.set('tile yes')

    for r in range(r0+1, r1+1):
        diffIm = ramp.cds(r0=r0, r1=r)
        disp.set('frame new')
        disp.set_np2arr(diffIm)

    disp.set('scale zscale')

def dispStackedVisits(disp, visits, cam, doClear=True, medSubtract=True,
                      r0=0, r1=-1):
    """Display the sum of a list of visit CDSes.

    Parameters
    ----------
    disp : DS9
        The DS9 display to write to
    visits : array of int
        The visits to load
    cam : `str`
        The name of the camera -- e.g. "n3"
    doClear : bool, optional
        Whether to delete all existing frames in the display, by default True
    medSubtract : bool, optional
        Whether to subract off the median background from all CDSes, by default True
    r0 : int, optional
        For each ramp, the first read of the CDS, by default 0
    r1 : int, optional
        For each ramp, the last read on the CDS, by default -1

    Returns
    -------
    image : np.array
        The data image which we displayed.
    """
    if doClear:
        disp.set('frame delete all')
        disp.set('frame lock image')
        disp.set('tile grid')
        disp.set('tile yes')

    stackedImage = None
    if isinstance(visits, int):
        visits = [visits]
    for v in visits:
        path = hx.rampPath(v, cam=cam)
        cds = hxramp.HxRamp(path).cdsN(r0=r0, r1=r1)
        if medSubtract:
            cds -= np.median(cds)
        if stackedImage is None:
            stackedImage = cds
        else:
            stackedImage += cds

    disp.set('frame new')
    disp.set_np2arr(stackedImage)

    return stackedImage

def getPix(row, meade, targetPos=False):
    """For a reading get the center pixel. Use measurement is available, else the steps. """

    try:
        xpix, ypix = row['xpix0'], row['ypix0']
    except:
        xpix, ypix = meade.stepsToPix([row['xstep'], row['ystep']])
    if targetPos:
        return xpix, ypix

    if 'xpix' in row and not np.isnan(row['xpix']):
        xpix, ypix = row['xpix'], row['ypix']
    return xpix, ypix

def dispMask(disp, alpha=0.5):
    disp.set(f'mask transparency {alpha*100}')
    disp.set(f'mask color red')
    disp.set(f'mask mark nonzero')

def setMask(disp, badMask, alpha=0.75):
    disp.set('mask clear')
    badMask = np.ascontiguousarray(badMask, dtype='i4')
    disp.set(f'array mask [xdim={badMask.shape[1]},ydim={badMask.shape[0]},bitpix=32]', badMask)
    dispMask(disp, alpha=alpha)

def tweakDisp(disp, zoom=None, center=None, tileGrid=None,
              lockImage=None, lockScale=None,
              scaleType=None, scaleLimits=None,
              doRun=True):

    ds9Cmds = []

    if tileGrid is not None:
        if tileGrid is False:
            ds9Cmds.append('tile no')
            ds9Cmds.append('tile grid mode automatic')
        else:
            if tileGrid is True:
                ds9Cmds.append('tile grid mode automatic')
            else:
                ds9Cmds.append(f'tile grid layout {tileGrid[0]} {tileGrid[1]}')
                ds9Cmds.append(f'tile grid direction x')
            ds9Cmds.append('tile mode grid')
            ds9Cmds.append('tile yes')

    if zoom is not None:
        ds9Cmds.append(f'zoom to {zoom}')
    if center is not None:
        ds9Cmds.append(f'pan to {center[0]} {center[1]}')

    if scaleType is not None:
        ds9Cmds.append(f'scale {scaleType}')
    if scaleLimits is not None:
        if isinstance(scaleLimits, str):
            ds9Cmds.append(f'scale mode {scaleLimits}')
        else:
            ds9Cmds.append(f'scale limits {scaleLimits[0]} {scaleLimits[1]}')

    if lockImage is not None:
        if lockImage:
            ds9Cmds.append('lock frame image')
        else:
            ds9Cmds.append('lock frame none')
    if lockScale is not None:
        if lockScale:
            ds9Cmds.append('lock frame scalelimits yes')
            ds9Cmds.append('lock frame scalelimits yes')
        else:
            ds9Cmds.append('lock frame scalelimits no')
            ds9Cmds.append('lock frame scale no')

    if doRun:
        disp.set('; '.join(ds9Cmds))
    else:
        return ds9Cmds

def setupDisp(disp, doClear=True, zoom=8, tileGrid=None,
              lockImage=True, lockScale=True,
              scaleType='asinh', scaleLimits=None):

    ds9Cmds = []
    if doClear:
        ds9Cmds.append('frame delete all')
        ds9Cmds.append('tile grid')
        ds9Cmds.append('tile yes')

    if tileGrid is None:
        tileGrid = True

    moreCmds = tweakDisp(disp, zoom=zoom, tileGrid=tileGrid,
                         lockImage=lockImage, lockScale=lockScale,
                         scaleType=scaleType, scaleLimits=scaleLimits,
                         doRun=False)
    ds9Cmds.extend(moreCmds)
    ds9CmdStr = '; '.join(ds9Cmds)
    disp.set(ds9CmdStr)

def dispSpots(disp, df, doClear=True, maxRows=16, tileGrid=None, r1=-1, meade=None, hxcalib=None,
              zoom=8, lockImage=False, lockScale=True, badMask=None, doOrder=True,
              scaleType='asinh', scaleLimits=None, ims=None, targetPos=False):
    """Show a grid of images centered on their spots."""

    if tileGrid is not None:
        nImages = tileGrid[0] * tileGrid[1]
        maxRows = nImages

    if len(df) > maxRows:
        raise ValueError(f"too many rows ({len(df)} > {maxRows}) -- increase maxRows if you really want")

    if doOrder:
        df = df.sort_values(by=['row', 'wavelength'],
                            ascending=[False, False])

    setupDisp(disp, doClear=doClear, zoom=zoom, tileGrid=tileGrid,
              lockImage=lockImage, lockScale=lockScale,
              scaleType=scaleType, scaleLimits=scaleLimits)

    for i_i, (i, row) in enumerate(df.iterrows()):
        disp.set('frame new')
        if ims is None:
            if hxcalib is not None:
                cds = hxcalib.isr(visit=row.visit, r1=r1)
            else:
                ramp = hxramp.HxRamp(pathUtils.rampPath(visit=row['visit'], cam=meade.cam))
                cds = ramp.cdsN(r1=r1)
            cds -= np.median(cds)
        else:
            cds = ims[i_i]
        disp.set_np2arr(cds)
        if badMask is not None:
            setMask(disp, badMask)
        if not lockImage:
            xpix, ypix = getPix(row, meade, targetPos=targetPos)
            disp.set(f'pan to {xpix} {ypix} image')
            logger.info(f'{row.wavelength:6.1f} {row["row"]:6.0f} {xpix:8.2f} {ypix:8.2f}')
        else:
            logger.info(f'{row.wavelength} {row.row}, {row.focus}')


    if lockImage: # Assume they are all at the same place
        row = df.head(1).squeeze()
        xpix, ypix = getPix(row, meade)
        disp.set(f'pan to {xpix} {ypix} image')
        disp.set('lock scalelimits; lock scale')

def ditherName(butler, group, pfsDay='*'):
    """Given a dither group, return the dither FITS file name. """
    focus = group.focus.unique()
    if len(focus) != 1:
        raise ValueError(f"not a unique focus value: {focus}")
    else:
        focus = focus[0]

    row = group.row.unique()
    if len(row) != 1:
        raise ValueError(f"not a unique row value: {row}")
    else:
        row = row[0]

    key = dict(focus=focus,
               wavelength=group.wavelength.unique()[0],
               visit=int(group.visit.min()),
               row=row)

    path = butler.search('dither', pfsDay=pfsDay, **key)
    return path[0]

def dispDithers(disp, butler, ditherSet, wavelength, zoom=8,
                scaleType='asinh', scaleLimits=None,
                badMask=None, pfsDay='*'):
    """Generate the canonical dither display: one page per wavelength, focus values per column."""

    waveDither = ditherSet[ditherSet.wavelength == wavelength].copy()
    waveDither = waveDither.sort_values(['row', 'focus'], ascending=[False, True])
    waveGroups = waveDither.groupby(['focus', 'row'], sort=False)

    nrows = len(waveDither.row.unique())
    nfocus = len(waveDither.focus.unique())

    setupDisp(disp, doClear=True, zoom=zoom, tileGrid=(nfocus, nrows),
              lockImage=False, lockScale=True,
              scaleType=scaleType, scaleLimits=scaleLimits)

    for name, group in waveGroups:
        path = ditherName(butler, group, pfsDay=pfsDay)
        print(name, path)

        im, hdr = fitsio.read(path, header=True)
        try:
            ctr = int(round(hdr['XPIX'])), int(round(hdr['YPIX']))
        except KeyError:
            ctr = [0,0]

        disp.set("frame new")
        if badMask is not None:
            rad = im.shape[0]//(3*2)
            xslice = slice(ctr[0]-rad, ctr[0]+rad)
            yslice = slice(ctr[1]-rad, ctr[1]+rad)

            imask = badMask[yslice, xslice]
            setMask(disp, imask)
        disp.set_np2arr(im)

    return waveGroups

def dispDithersAtFocus(disp, butler, ditherSet, focus, zoom=8,
                       scaleType='asinh', scaleLimits=None,
                       badMask=None, pfsDay='*'):
    """Generate a single-focus dither display"""

    focusDither = ditherSet[ditherSet.focus == focus].copy()
    focusDither = focusDither.sort_values(['row', 'wavelength'], ascending=[False, False])
    waveGroups = focusDither.groupby(['wavelength', 'row'], sort=False)

    nrows = len(focusDither.row.unique())
    nwaves = len(focusDither.wavelength.unique())

    setupDisp(disp, zoom=zoom, tileGrid=(nwaves, nrows),
              lockImage=False, lockScale=True,
              scaleType=scaleType, scaleLimits=scaleLimits)

    for name, group in waveGroups:
        path = ditherName(butler, group, pfsDay=pfsDay)
        print(name, path)

        im, hdr = fitsio.read(path, header=True)
        try:
            ctr = int(round(hdr['XPIX'])), int(round(hdr['YPIX']))
            rad = im.shape[0]//(3*2)
        except KeyError:
            ctr = im.shape[1]//2, im.shape[0]//2
            rad = min(im.shape[0], im.shape[1])//2
        xslice = slice(ctr[0]-rad, ctr[0]+rad)
        yslice = slice(ctr[1]-rad, ctr[1]+rad)

        disp.set("frame new")
        if badMask is not None:
            imask = badMask[yslice, xslice]
            setMask(disp, imask)
        disp.set_np2arr(im)

    return waveGroups

def dispDitherDetails(disp, butler, ditherSet):
    setupDisp(disp)

def ditherDiag1(df):
    """Show the measured positions of all individual dither spots. """
    names, xspans, yspans = nirander.ditherScales(df)
    
    f,pl = plt.subplots(ncols=2, clear=True)
    p1, p2 = pl
    p1.set_aspect('equal')
    xr = []
    yr = []
    for x_i, x in enumerate(xspans): 
        y = yspans[x_i]
        x -= x[0,0]
        y -= y[0,0]
        xd = np.mean(x[:,-1] - x[:,0])
        yd = np.mean(y[-1,:] - y[0,:])
        xr.append(xd)
        yr.append(yd)

        p1.plot(x,y, '+', alpha=0.3, markersize=2)
    p1.set_xlim(-0.25, 1.0)
    p1.set_ylim(-0.25, 1.0)

    p1.hlines([0,0.33,0.66], xmin=-0.25, xmax=1, alpha=0.5, color='k', linewidth=0.2)
    p1.vlines([0,0.33,0.66], ymin=-0.25, ymax=1, alpha=0.5, color='k', linewidth=0.2)

    xr = np.array(xr)
    yr = np.array(yr)
    p2.hist(xr, alpha=0.3, bins=20, range=(0.25, 1.0), label=f'x {np.nanmean(xr):0.2f} +/- {np.nanstd(xr):0.2f}')
    p2.hist(yr, alpha=0.3, bins=20, range=(0.25, 1.0), label=f'y {np.nanmean(yr):0.2f} +/- {np.nanstd(yr):0.2f}')

    p2.legend(fontsize='x-small')
    print(np.nanmean(xr), np.nanstd(xr))
    print(np.nanmean(yr), np.nanstd(yr))

    f.suptitle(f'{len(names)} dithers, visit={df.visit.min()}..{df.visit.max()}')
    f.tight_layout()
    return f

def dispTransform(df, meade=None, scale=1.0, title=''):
    # updateTargets(df, meade)
    try:
        plt.close('xform')
    except:
        pass
    f, pl = plt.subplots(num='xform', clear=True)

    if meade is not None:
        xpix0 = np.empty(shape=len(df))
        ypix0 = np.empty(shape=len(df))
        for row_i, row in df.reset_index().iterrows():
            xpix, ypix = meade.stepsToPix([row.xstep, row.ystep])
            xpix0[row_i] = xpix
            ypix0[row_i] = ypix
    else:
        xpix0 = df.xpix0
        ypix0 = df.ypix0
        
    q = pl.quiver(xpix0, ypix0, df.xpix-xpix0, df.ypix-ypix0,
                  width=0.003) # , scale_units='xy', scale=10)
    pl.set_ylim(-500,4600)
    pl.set_xlim(-500,4600)
    pl.quiverkey(q, X=0.1, Y=1.05, U=scale, label=f'{scale:0.0f} pix resid', labelpos='E', alpha=0.6)
    f.suptitle(f'Gimbalator step->pix visit={df.visit.min()} {title}')

    return f
