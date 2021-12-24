from importlib import reload
import logging
import os
import pathlib

import fitsio
import numpy as np
import matplotlib.pyplot as plt

import sep

from . import nirander
from . import hxstack as hx
from . import hxramp

reload(hx)
reload(hxramp)
reload(nirander)

logger = logging.getLogger('hxdisplay')
logger.setLevel(logging.INFO)

def camFromPath(fname):
    p = pathlib.Path(fname)
    return 'n' + p.stem[-2]

def regStats(fpath, slices=None, r0=0, r1=-1, fitr0=1, fitr1=-1, doDiffs=False, 
             residMax=-1, order=1, preampGain=None):
    
    ff = hx.ramp(fpath)
    nreads = hx.rampNreads(ff)
    height, width = ff[1].read().shape
    
    readTime = (width / 4096) * hx.singleReadTime
    if preampGain is not None:
        adcGain = hx.calcGain(preampGain, camFromPath(fpath))
    else:
        preampGain = 1.0
        adcGain = 1.0
    
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
        plotReads = np.arange(r0, nreads)
    else:
        plotReads = np.arange(r0, r1+1)
    print(plotReads)
    for r_i in plotReads:
        im = hx.rampRead(ff, r_i) * adcGain
        reg = im[slices]
        meds.append(np.median(reg))
        if r_i > r0:
            lastReg = hx.rampRead(ff, r_i - 1)[slices] * adcGain
            dim = reg - lastReg
            dmeds.append(np.median(dim))
            ddevs.append(np.std(dim))
    dmeds = np.array(dmeds)
    ddevs = np.array(ddevs)
    yoffset = meds[fitr0]
    
    plotX = plotReads * readTime
    fitY = np.array(meds[fitr0:fitr1])
    fitX = plotX[fitr0:fitr1]
    coeffs = np.polyfit(fitX, fitY, order)
    print(f'x: {fitX}')
    print(f'y: {fitY}')
    
    label = "%0.3ft + %0.2f" % (coeffs[0], coeffs[1])
    line = np.poly1d(coeffs)
    # print(len(meds), len(fitX), len(fitY))
    
    p1.plot(plotX, meds - coeffs[-1], '+-')
    p1.plot(plotX, line(plotX) - coeffs[-1], ':', color='red', label=label)
    p1.plot(fitX, line(fitX) - coeffs[-1], color='red')

    residX = plotX
    residY = meds
    if residMax is not None:
        residX = plotX[:residMax]
        residY = meds[:residMax]
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

def focusPlot(sweep, p=None, title=None):
    if p is None:
        f,p1 = plt.subplots(num='focusPlot', clear=True)
    else:
        f = None
        p1 = p
    
    if title is None:
        title = f"{sweep['wavelength'].values[0]:0.0f} @ {sweep['row'].values[0]:0.0f}"
        
    sweep = sweep.sort_values(by=['focus'])

    minx, poly = nirander.getBestFocus(sweep)
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
        
    return f, minx, poly(minx)

def focusOffsetPlot(meade, focusResults, visit=None):
    f, pl = plt.subplots(num='focusOffsetGrid', clear=True)
    x, y = meade.stepsToPix(focusResults['xstep'], focusResults['ystep'])
    val = pl.scatter(x.values, y.values, c=focusResults['focus'].values, cmap=plt.cm.get_cmap('viridis'))
    f.colorbar(val)
    f.suptitle(f'best focus, from visit={visit}')
    return f

def dispVisits(disp, visits, r0=0, r1=-1, cam='n1', doClear=True):
    if doClear:
        disp.set('frame delete all')
    disp.set('frame lock image')
    disp.set('tile grid')
    disp.set('tile yes')

    if isinstance(visits, int):
        visits = [visits]
    for v in visits:
        path = hx.rampPath(v, cam=cam)
        cds = hxramp.HxRamp(path).cdsN(r0=r0, r1=r1)
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

def dispStackedVisits(disp, visits, cam, doClear=True, r0=0, r1=-1):
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
        if stackedImage is None:
            stackedImage = cds
        else:
            stackedImage += cds

    disp.set('frame new')
    disp.set_np2arr(stackedImage)

def getPix(row, meade):
    """For a reading get the center pixel. Use measurement is available, else the steps. """
    try:
        xpix, ypix = row['xpix'], row['ypix']
        if np.isnan(xpix):
            raise ValueError()
    except:
        xpix, ypix = meade.stepsToPix([row['xstep'], row['ystep']])

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

def dispSpots(disp, df, doClear=True, maxRows=16, tileGrid=None, r1=-1, meade=None,
              zoom=8, cam=None, lock=True, badMask=None, doOrder=True,
              scaleType='asinh', scaleLimits=None):
    """Show a grid of images centered on their spots."""

    if tileGrid is not None:
        nImages = tileGrid[0] * tileGrid[1]
        maxRows = nImages

    if len(df) > maxRows:
        raise ValueError(f"too many rows ({len(df)} > {maxRows}) -- increase maxRows if you really want")

    if doOrder:
        df = df.sort_values(by=['row', 'wavelength'],
                            ascending=[False, True])
    if doClear:
        disp.set('frame delete all')
        disp.set('tile grid')
        disp.set('tile yes')
        disp.set(f'zoom to {zoom}')
        disp.set(f'scale {scaleType}')
        if scaleLimits is not None:
            if isinstance(scaleLimits, str):
                disp.set(f'scale mode {scaleLimits}')
            else:
                disp.set(f'scale limits {scaleLimits[0]} {scaleLimits[1]}')
    if tileGrid is not None:
        disp.set(f'tile grid layout {tileGrid[0]} {tileGrid[1]}')
        disp.set(f'tile grid direction x')
    else:
        disp.set('tile grid mode automatic')

    if lock:
        disp.set('lock frame image')
    else:
        disp.set('lock frame none')

    for i, row in df.iterrows():
        disp.set('frame new')
        ramp = hxramp.HxRamp(hxramp.rampPath(visit=row['visit'], cam=cam))
        cds = ramp.cdsN(r1=r1)
        cds -= np.median(cds)
        disp.set_np2arr(cds)
        if badMask is not None:
            setMask(disp, badMask)
        if not lock:
            xpix, ypix = getPix(row, meade)
            disp.set(f'pan to {xpix} {ypix} image')
            logger.info(f'{row.wavelength} {row["row"]:0.1f} {xpix:0.2f} {ypix:0.2f}')

    if lock: # Assume they are all at the same place
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
               wave=group.wavelength.unique()[0],
               visit=group.visit.min(),
               row=row)

    path = butler.search('dither', pfsDay=pfsDay, **key)
    return path[0]

def dispDithers(disp, butler, ditherSet, wavelength, zoom=8, scale=99.7, scaleType='linear',
                badMask=None, pfsDay='*'):
    """Generate the canonical dither display: one page per wavelength, focus values per column."""

    waveDither = ditherSet[ditherSet.wavelength == wavelength].copy()
    waveDither = waveDither.sort_values(['row', 'focus'], ascending=[False, True])
    waveGroups = waveDither.groupby(['focus', 'row'], sort=False)

    nrows = len(waveDither.row.unique())
    nfocus = len(waveDither.focus.unique())

    disp.set('frame delete all')
    disp.set('tile grid')
    disp.set('tile yes')
    disp.set(f'zoom to {zoom}')
    disp.set(f'scale {scaleType}')
    disp.set(f'scale mode {scale}')
    disp.set(f'tile grid layout {nfocus} {nrows}')
    disp.set('tile grid direction x')

    disp.set('lock frame none')
    disp.set('lock frame scalelimits yes')

    for name, group in waveGroups:
        path = ditherName(butler, group, pfsDay=pfsDay)
        print(name, path)

        im, hdr = fitsio.read(path, header=True)
        ctr = int(round(hdr['XPIX'])), int(round(hdr['YPIX']))
        rad = im.shape[0]//(3*2)
        xslice = slice(ctr[0]-rad, ctr[0]+rad)
        yslice = slice(ctr[1]-rad, ctr[1]+rad)

        disp.set("frame new")
        if badMask is not None:
            imask = badMask[yslice, xslice]
            setMask(disp, imask)
        disp.set_np2arr(im)

    return waveGroups

def dispDithersAtFocus(disp, butler, ditherSet, focus, zoom=8, scale=99.7, scaleType='linear',
                badMask=None, pfsDay='*'):
    """Generate a single-focus dither display"""

    focusDither = ditherSet[ditherSet.focus == focus].copy()
    focusDither = focusDither.sort_values(['row', 'wavelength'], ascending=[False, False])
    waveGroups = focusDither.groupby(['wavelength', 'row'], sort=False)

    nrows = len(focusDither.row.unique())
    nwaves = len(focusDither.wavelength.unique())

    disp.set('frame delete all')
    disp.set('tile grid')
    disp.set('tile yes')
    disp.set(f'zoom to {zoom}')
    disp.set(f'scale {scaleType}')
    disp.set(f'scale mode {scale}')
    disp.set(f'tile grid layout {nwaves} {nrows}')
    disp.set('tile grid direction x')

    disp.set('lock frame none')
    disp.set('lock frame scalelimits yes')

    for name, group in waveGroups:
        path = ditherName(butler, group, pfsDay=pfsDay)
        print(name, path)

        im, hdr = fitsio.read(path, header=True)
        ctr = int(round(hdr['XPIX'])), int(round(hdr['YPIX']))
        rad = im.shape[0]//(3*2)
        xslice = slice(ctr[0]-rad, ctr[0]+rad)
        yslice = slice(ctr[1]-rad, ctr[1]+rad)

        disp.set("frame new")
        if badMask is not None:
            imask = badMask[yslice, xslice]
            setMask(disp, imask)
        disp.set_np2arr(im)

    return waveGroups