from importlib import reload
import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt

import sep

import nirander
import hxstack as hx
reload(hx)
reload(nirander)

def camFromPath(fname):
    p = pathlib.Path(fname)
    return 'n' + p.stem[-2]

def regStats(fpath, slices=None, r0=0, r1=-1, fitr0=1, fitr1=-1, doDiffs=False, 
             order=1, preampGain=None):
    
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
    p1.plot(fitX, line(fitX) - coeffs[-1], label=label)
    p2.plot(fitX, fitY - line(fitX), label='residuals')
    p2.hlines(0,fitX[0],fitX[-1],colors='r',alpha=0.3)
    title = "%s\nx=[%d,%d] y=[%d,%d]  nreads=%d %0.2f $e^-/ADU$ preampGain=%0.2f readTime=%0.2f" % (os.path.basename(fpath), 
                                                                                                    slices[1].start,
                                                                                                    slices[1].stop-1, 
                                                                                                    slices[0].start,
                                                                                                    slices[0].stop-1,
                                                                                                    nreads, adcGain,
                                                                                                    preampGain, readTime)
    p1.set_title(title)
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
        title = f"{sweep['wavelength'].values[0]:0.0f} ({sweep['xpix'].values[0]:0.0f},{sweep['ypix'].values[0]:0.0f})"
        
    sweep = sweep.sort_values(by=['focus'])

    minx, poly = nirander.getBestFocus(sweep)
    p1.plot(sweep.focus, sweep.x2, '+-', alpha=0.3, label='x')
    p1.plot(sweep.focus, sweep.y2, '+-', alpha=0.3, label='y')
    p1.plot(sweep.focus, sweep['size'], '+-', alpha=0.75, label='(x+y)/2')
    
    xx = np.linspace(sweep.focus.min(), sweep.focus.max()+1, 100)
    yy = poly(xx)
    # print(f'{title}: {minx:0.2f}, {poly(minx):0.2f}')
    p1.plot(xx, yy, label=f'best={minx:0.0f}')
    p1.plot([minx], [poly(minx)], 'kx', markersize=10)
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
