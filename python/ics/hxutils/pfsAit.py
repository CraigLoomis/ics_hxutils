import functools
from importlib import reload
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
import fitsio

from pfs.utils import butler as pfsButler

from . import hxramp
from . import pfsutils
from . import nirander
from . import hxdisplay
from . import mplutils

reload(pfsButler)
reload(hxramp)
reload(pfsutils)
reload(nirander)
reload(hxdisplay)

logger = logging.getLogger('pfsAit')
logger.setLevel(logging.INFO)

@functools.lru_cache(20000)
def genLagrangeCoeffs(xshift, order=4):
    """ Return Lagrange coefficients for the #order points centered on the requested shift.

    Args
    ----
    xshift : float
      How much to shift by. Must be (-1..0) or (0..1)
    order : integer
      The order of the Lagrange polynomial

    Returns
    -------
    outSlice : slice
      The slice to save the interpolated sum to
    xSlices : list of slices, len=order
      The per-y slices for the inputs
    coeffs : list of floats, len=order
      The coefficients to multiply the sliced data by
    """

    if xshift == 0:
        raise ValueError('xshift must be non-0.')
    if abs(xshift) >= 1:
        raise ValueError('abs(xshift) must be less than 1.')

    # We are trying to center the output between input points.
    if xshift > 0:
        x = 1 - xshift
        xp = range(-order//2+1, order//2+1)
        xlo = range(0, order)
        xhi = range(-order, 0)
    else:
        x = -1 - xshift
        xp = range(-order//2, order//2)
        xlo = range(1, order+1)
        xhi = range(-(order-1), 0)
        xhi = list(xhi) + [None]

    outSlice = slice(order//2,-order//2)

    xSlices = []
    coeffs = []
    for c_i, cn in enumerate(xp):
        num = 1.0
        denum = 1.0

        for x_i, xn in enumerate(xp):
            if x_i == c_i:
                continue
            num *= (x-xn)
            denum *= (c_i-x_i)

        coeffs.append(num/denum)
        xSlices.append(slice(xlo[c_i], xhi[c_i]))

    return outSlice, xSlices, coeffs

def _clipOnes(d, precision):
    if d == 1.0:
        return d - 1/precision
    elif d == -1.0:
        return d + 1/precision
    else:
        return d

def shiftSpotLagrange(img, dx, dy, order=4, kargs=None, precision=100):
    """ Shift a spot using order=4 Lagrange interpolation.

    Args
    ----
    img : 2-d image
      The spot image to shift. Assumed to have enough border to do so.
    dx, dy : float
      How much to shift the spot in each direction. should be (-1..1)
    order : integer
      The Lagrange polynomial order. 4 gives two input points on each side.
    kargs : dict
      Unused, declared to be compatible with other shift functions.
    precision : int
      Inverse precision to allow for dx,dy; lower values provide more
      cache hits.

    Returns
    -------
    outImg : the shifted spot
    None   : compatibility turd.

    Notes
    -----

    It turns out that:
      out = img[slice0]
      out += img[slice1]
      out += img[slice2]

    is _significantly_ slower than:
      out = img[slice0] + img[slice1] + img[slice2]

    Hence the eval string.
    """

    dx = int(dx*precision + 0.5)/precision
    dy = int(dy*precision + 0.5)/precision
    dx = _clipOnes(dx, precision)
    dy = _clipOnes(dy, precision)

    if abs(dx) < 1e-6:
        outImg1 = img
    else:
        outSlice, xSlices, coeffs = genLagrangeCoeffs(dx)

        outImg1 = np.zeros(shape=img.shape, dtype=img.dtype)
        for ii in range(order):
            outImg1[:, outSlice] += coeffs[ii]*img[:, xSlices[ii]]

    if abs(dy) < 1e-6:
        outImg = outImg1
    else:
        outSlice, ySlices, coeffs = genLagrangeCoeffs(dy)

        outImg = np.zeros(shape=img.shape, dtype=img.dtype)
        for ii in range(order):
            outImg[outSlice, :] += coeffs[ii]*outImg1[ySlices[ii], :]

    return outImg, None

shiftSpot = shiftSpotLagrange

def groupDithers(ditherSet, groupby=None):
    if groupby is None:
        groupby = ['wavelength', 'row', 'focus']
    ditherGroups = ditherSet.groupby(groupby)
    dig = ditherGroups['visit'].aggregate(np.min).reset_index()
    ret = ditherSet[ditherSet.visit.isin(dig.visit)]

    return ret

def groupAllDithers(ditherSet, groupby=None):
    res = []

    if groupby is None:
        groupby = ['wavelength', 'row', 'focus']
    ditherGroups = ditherSet.groupby(groupby)
    for dname, grp in ditherGroups:
        minx = grp.xstep.min()
        miny = grp.ystep.min()
        llStep = ditherSet.loc[(ditherSet.xstep == minx) & (ditherSet.ystep == miny)]
        res.append(llStep)
    return pd.concat(res, ignore_index=True)

def ditherFromVisit(ditherSet, visit):
    ditherSet = ditherSet.sort_values('visit', ignore_index=True)
    dith = ditherSet.loc[(ditherSet.visit >= visit) & (ditherSet.visit <= visit+9)]
    return dith

def selectDithers(rows, *, wavelength=None, row=None, focus=None):
    dithers = groupDithers(rows)
    if wavelength is not None:
        dithers = dithers.loc[dithers.wavelength == wavelength]
    if row is not None:
        dithers = dithers.loc[dithers.row == row]
    if focus is not None:
        dithers = dithers.loc[dithers.focus == focus]

    return dithers.reset_index(drop=True)

def measureDithers(butler, rows, thresh=50,
                   radius=30, searchRadius=10,
                   hxcalib=None, pfsDay='*'):
    """Center up and measure dithers

    Parameters
    ----------
    butler : `butler.Butler`
        How to get ands save data
    rows : `pd.DataFrame`
        a DataFrame for the *dithers*, not the spots.
    thresh : int, optional
        Detection threshold, by default 250. Should be restated in terms of bg sigma.
    radius : int, optional
        The size of the patch we measure, by default 30 5um pixels
    searchRadius : int, optional
        How far we look to find a peak, by default 10 5um pixels
    hxcalib : `hxramp.HxCalib`, optional
        Used for basic ISR, by default None
    pfsDay : str, optional
        Where the butler should search for dithers, by default '*'

    Returns
    -------
    meas : `pd.DataFrame`
        The measurements of the centered spots.
    centeredSpots : image
        The spot images after re-centering.
    rawSpots : image
        The spot images before re-centering
    """
    rawDithers = []
    centeredDithers = []
    centeredPeaks = []
    for r_i in range(len(rows)):
        row = rows.iloc[r_i:r_i+1].copy()
        path = nirander.ditherPath(butler, row, pfsDay=pfsDay)
        dither, hdr = fitsio.read(path, header=True)
        rawDithers.append(dither)

        # Hmm. Assumes that the dither is basically centered already. But
        # if we cannot depend on the gimbelator, that may not be right.
        ditherCtr = (np.array(dither.shape)[::-1] + 1) // 2
        ctrX = ditherCtr[1]
        ctrY = ditherCtr[0]
        meas = nirander.measureSet(row, center=ditherCtr,
                                   radius=radius, searchRadius=searchRadius,
                                   ims=[dither],
                                   hxCalib=hxcalib,  thresh=thresh, doClear=True)
        if len(meas) != 1:
            raise RuntimeError(f"{len(meas)} peaks for {path}: {meas}")
        if np.isnan(meas.xpix.values[0]):
            logger.warning(f"peak for {path} not measured")
            centeredDithers.append(dither)
            centeredPeaks.append(meas)
            continue

        measX = meas.xpix.values[0]
        measY = meas.ypix.values[0]
        dx = ditherCtr[0] - measX
        dy = ditherCtr[1] - measY

        if abs(dx) >= 1:
            pixDist = int(abs(dx))
            if dx < 0:
                dither[:, :-pixDist] = dither[:, pixDist:]
                dither[:, -pixDist:] = np.median(dither[:, -pixDist:])
                dx += pixDist
            else:
                dither[:, pixDist:] = dither[:, :-pixDist]
                dither[:, :pixDist] = np.median(dither[:, :pixDist])
                dx -= pixDist
        if abs(dy) >= 1:
            pixDist = int(abs(dy))
            if dy < 0:
                dither[:-pixDist, :] = dither[pixDist:, :]
                dither[-pixDist:, :] = np.median(dither[-pixDist:, :])
                dy += pixDist
            else:
                dither[pixDist:, :] = dither[:-pixDist, :]
                dither[:pixDist, :] = np.median(dither[:pixDist, :])
                dy -= pixDist

        centeredDither = shiftSpot(dither, dx, dy)[0]
        peaks = nirander.measureSet(meas, center=ditherCtr,
                                    radius=radius, searchRadius=searchRadius,
                                    ims=[centeredDither],
                                    hxCalib=hxcalib, thresh=thresh, doClear=True)
        # _, peaks = nirander.getPeaks(dith, center=ctr, radius=5)
        if len(peaks) != 1:
            raise RuntimeError(f"{len(peaks)} peaks for {path}: {peaks}")
        if np.isnan(peaks.xpix.values[0]):
            logger.warning(f"peaks for {path} not measured")
        else:
            peaks['size'] *= 5
            ee1 = centeredDither[ctrY, ctrX]
            if peaks['size'].values[0] < 8: # Avoid checking donuts.
                if ee1 < centeredDither[ctrY-1, ctrX] or ee1 < centeredDither[ctrY+1, ctrX] or ee1 < centeredDither[ctrY, ctrX-1] or ee1 < centeredDither[ctrY, ctrX+1]:
                    logger.debug(f"ee1 pixel is less than some neighbor: {meas.wavelength.values[0]} @ {meas.row.values[0]}\n"
                                   f"{peaks}\n"
                                   f"{centeredDither[ctrY-1:ctrY+2,ctrX-1:ctrX+2]}")
            ee3 = centeredDither[ctrY-1:ctrY+2, ctrX-1:ctrX+2].sum()
            ee5 = centeredDither[ctrY-2:ctrY+3, ctrX-2:ctrX+3].sum()
            peaks['ee1'] = ee1 / peaks.flux.values[0]
            peaks['ee3'] = ee3 / peaks.flux.values[0]
            peaks['ee5'] = ee5 / peaks.flux.values[0]

            nirander.writeDither(peaks, butler, centeredDither)

        centeredDithers.append(centeredDither)
        centeredPeaks.append(peaks)

    return pd.concat(centeredPeaks), centeredDithers, rawDithers

def fetchFocusPlane(df):

    ret = []
    for w_i, w in enumerate(sorted(df.wavelength.unique())[::-1]):
        for r_i, r in enumerate(sorted(df.row.unique())):
            frows = df.loc[(df.wavelength == w) & (df.row == r)]
            fmin, fcoeffs = nirander.getBestFocus(frows)

            ret.append((r, w, fmin, fcoeffs))

    focusFrame = pd.DataFrame(ret, columns=['row', 'wave', 'focus', 'coeffs'])
    return focusFrame

def fetchBestFocusPlane(df, name):
    ret = []
    for w_i, w in enumerate(sorted(df.wavelength.unique())[::-1]):
        for r_i, r in enumerate(sorted(df.row.unique())):
            frows = df.loc[(df.wavelength == w) & (df.row == r)]
            fmin, fcoeffs = nirander.getPolyMin(frows, 'focus', name)

            ret.append((r, w, fmin, fcoeffs))

    focusFrame = pd.DataFrame(ret, columns=['row', 'wave', 'focus', 'coeffs'])
    return focusFrame

def fetchLimitPlane(df, name):
    ret = []

    grps = df.gro
    for w_i, w in enumerate(sorted(df.wavelength.unique())[::-1]):
        for r_i, r in enumerate(sorted(df.row.unique())):
            frows = df.loc[(df.wavelength == w) & (df.row == r)]
            maxVal = frows.loc[:, name].max()
            minVal = frows.loc[:, name].min()

            ret.append((r, w, minVal, maxVal))

    focusFrame = pd.DataFrame(ret, columns=['row', 'wave', 'focus', 'coeffs'])
    return focusFrame

def focusColor(focus, center, limits):
    """[Return a color from a colormap hinged at a center value.

    Args:
        focus (float): the focus value we want a color for
        centerFocus (float): the best focus around which the colormap is centered
        focusRange ([float, float[]]): the range of focus
    Returns:
        float: colormap value.
    """

    cmap = plt.get_cmap('seismic')
    minVal, maxVal = limits
    fullScale = maxVal - minVal
    dFocus = (focus - center)
    focusFrac = dFocus / fullScale + 0.5

    color = cmap(focusFrac)
    return color

def normalize(vals, center=None, fullRange=None):
    if fullRange is None:
        fullRange = vals.min(), vals.max()
    minVal, maxVal = fullRange
    fullScale = maxVal - minVal

    frac = (vals - minVal) / fullScale

    return frac

def dispFocusPlane0(df, focusCenter=None, pl=None):
    if pl is None:
        f, pl = plt.subplots(figsize=(8,8))
    else:
        f = pl.figure
    # pl.set_aspect('equal')

    focusGrid = fetchFocusPlane(df)
    focusRange = (focusGrid.focus.min(), focusGrid.focus.max())
    if focusCenter is None:
        focusCenter =  focusGrid.focus.mean()
    focusObj = []
    for row in focusGrid.itertuples():
        o = pl.plot(row.wave, row.row, 'o', markersize=20,
                    color=focusColor(row.focus, focusCenter, focusRange))
        focusObj.append(o)

    # plt.colorbar(o)

    #pl.set_xbound(df.wavelength.min(), df.wavelength.max())

    return f

def dispFocusPlane(df, focusCenter=None, focusRange=None, pl=None, cmapName='RdBu'):
    if pl is None:
        f, pl = plt.subplots(figsize=(8,8))
    else:
        f = pl.figure
    # pl.set_aspect('equal')

    focusGrid = fetchFocusPlane(df)
    if focusRange is None:
        focusRange = (focusGrid.focus.min(), focusGrid.focus.max())
    if focusCenter is None:
        focusCenter = focusGrid.focus.mean()
    focusObj = []

    if focusCenter <= focusRange[0]:
        focusCenter = focusRange[0] + 1
    elif focusCenter >= focusRange[1]:
        focusCenter = focusRange[1] - 1

    norm = mpl.colors.TwoSlopeNorm(vcenter=focusCenter,
                                   vmin=focusRange[0], vmax=focusRange[1])
    cmap = plt.get_cmap(cmapName)
    colors = focusColor(focusGrid.focus, focusCenter, focusRange)
    o = pl.scatter(focusGrid.wave, focusGrid.row, c=focusGrid.focus, marker='o', s=300,
                   norm=norm, cmap=cmap)
    plt.colorbar(o, ax=pl, fraction=0.1, pad=0.01)
    xticks = mpl.ticker.FixedLocator(df.wavelength.unique())
    yticks = mpl.ticker.FixedLocator(df.row.unique())
    pl.xaxis.set_major_locator(xticks)
    pl.yaxis.set_major_locator(yticks)

    pl.set_xlabel('wavelength')
    pl.set_title('best focus (um)')
    #pl.set_xbound(df.wavelength.min(), df.wavelength.max())

    return f

def dispPlane(df, name, req=None, plotRange=None, pl=None,
              markersize=300, cmapName='seismic', label='mean'):
    if pl is None:
        f, pl = plt.subplots(figsize=(10,8))
    else:
        f = pl.figure
    # pl.set_aspect('equal')

    if req is None:
        req = df[name].mean()

    if plotRange is None:
        plotRange = [df[name].min(), df[name].max()]
    if plotRange[1] is None:
        plotRange[1] = req

    if plotRange[0] > plotRange[1]:
        plotRange[0], plotRange[1] = plotRange[1], plotRange[0]
        cmap = plt.get_cmap(cmapName + '_r')
        normVals = df[name]/req
    else:
        cmap = plt.get_cmap(cmapName)
        normVals = req/df[name]

    norm = mpl.colors.TwoSlopeNorm(vcenter=req, vmin=plotRange[0], vmax=plotRange[1])
    o = pl.scatter(df.wave, df.row, s=markersize*normVals, cmap=cmap, norm=norm,
                   c=df[name],
                   marker='o')
    xticks = mpl.ticker.FixedLocator(df.wave.unique())
    yticks = mpl.ticker.FixedLocator(df.row.unique())
    pl.xaxis.set_major_locator(xticks)
    pl.yaxis.set_major_locator(yticks)
    pl.set_xlabel('wavelength')
    pl.set_ylabel('row')
    pl.set_title(f'{name}, {label}={df[name].mean():0.3f}')

    cb = plt.colorbar(o, ax=pl, fraction=0.1, pad=0.01, format='%0.3f')
    cb.ax.axhline(y=req, c='k')

    original_ticks = list(cb.get_ticks())
    cb.set_ticks(original_ticks + [req])
    # cb.set_ticklabels(original_ticks + [f'{req:0.3f}'])

    #pl.set_xbound(df.wavelength.min(), df.wavelength.max())

    return f

def dispSizes(df, atFocus=None, focusRange=None, title=None):
    f, pl = plt.subplots(nrows=2, ncols=2, num='sizes',
                         clear=True, figsize=(10,10),
                         sharex=True, sharey=True, squeeze=False)
    pl = pl.flatten()

    # Note: due to sharex=True, do this here once. It will apply to all plots.
    pl[0].invert_xaxis()

    if atFocus is None:
        spotFrame = df
    else:
        spotFrame = df.loc[df.focus == atFocus]
    grps = spotFrame.sort_values(['wavelength', 'row'],
                                 ascending=[False, False]).groupby(['wavelength', 'row'],
                                                                   sort=False)
    mm = []
    for name,grp in grps:
        mm.append((name[0], name[1], grp.ee1.max(), grp.ee3.max(), grp.ee5.max(),
                   grp['size'].min()))
    spotees = pd.DataFrame(mm, columns=['wave', 'row', 'ee1', 'ee3', 'ee5', 'size'])
    spotees['focus'] = atFocus

    def _makePlotRange(name, req, under=0.5, over=1.1):
        return (name, req, [req*under, req*over])

    plots = (_makePlotRange('ee1', 0.085),
             _makePlotRange('ee3', 0.544),
             _makePlotRange('ee5', 0.907),
             _makePlotRange('size', 18.9, under=1.5, over=0.9))
    if atFocus is None:
        label = "mean of BEST"
    else:
        label = "mean"

    for p_i, (name, requirement, plotRange) in enumerate(plots):
        dispPlane(spotees, name, req=requirement, plotRange=plotRange,
                  pl=pl[p_i], label=label)
    # dispFocusPlane(df, focusCenter=atFocus, focusRange=focusRange, pl=pl[-1])

    # Hack-y cleanup
    pl[0].set_xlabel('')
    pl[1].set_xlabel('')

    pl[1].set(xlabel='')
    pl[1].set_ylabel('')

    if title is None:
        if atFocus is None:
            atFocus = "BEST FOR EACH"
        title = f'dithers from visit={df.visit.min()} at focus={atFocus}'
    f.suptitle(title, size='x-large')
    # f.tight_layout()

    return f

def dispEEPlane(df, name, focusCenter=None):
    f, pl = plt.subplots(figsize=(8,8))
    # pl.set_aspect('equal')

    focusGrid = fetchBestFocusPlane(df, name)

    focusRange = (focusGrid[name].min(), focusGrid[name].max())
    if focusCenter is None:
        focusCenter =  focusGrid[name].mean()
    focusObj = []
    for row in focusGrid.itertuples():
        logger.info(f'row={row}')
        o = pl.plot(row.wave, row.row, 'o', markersize=20,
                    color=focusColor(row[name], focusCenter, focusRange))
        focusObj.append(o)

    # plt.colorbar(ax=pl)

    #pl.set_xbound(df.wavelength.min(), df.wavelength.max())

    return f

def dispFocusPlots(df, title=None, yrange=None):
    nrows = len(df.row.unique())
    ncols = len(df.wavelength.unique())

    f, pl = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,
                         figsize=(10,10), squeeze=False)

    for r_i in range(nrows):
        pl[r_i,0].set_ylabel('spot size (um)')
    for c_i in range(ncols):
        pl[-1,c_i].set_xlabel('focus pos (um)')


    for w_i, w in enumerate(sorted(df.wavelength.unique())[::-1]):
        for r_i, r in enumerate(sorted(df.row.unique())[::-1]):
            frows = df.loc[(df.wavelength == w) & (df.row == r)]
            hxdisplay.focusPlot(frows, pl[r_i][w_i], sizeOnly=True,
                                dithers=True, yrange=yrange)

    finalTitle=f'dithers {df.visit.min()}..{df.visit.max()}'
    if title is not None:
        finalTitle = f'{finalTitle} {title}'
    f.suptitle(finalTitle)
    f.tight_layout()

    return f

def dispOffsets(df, title=None, yrange=None, focus=None, perSpot=True,
                figsize=(10,10)):
    """Diagnostic plots for dither repeats

    Parameters
    ----------
    df : DataFrame
        The measured positions
    title : str, optional
        plot title, by default something visit-based
    yrange : tuple, optional
        The height of the plots in pixels, by default None
    focus : int, optional
        Which focus to select, by default None
    perSpot : bool, optional
        For multiples, whether to register to LL corner, by default True

    Returns
    -------
    figure : matplotlib.Figure
    """
    nrows = len(df.row.unique())
    ncols = len(df.wavelength.unique())

    f, pl = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,
                         figsize=figsize, squeeze=False)

    for r_i in range(nrows):
        pl[r_i,0].set_ylabel('dither offset (15um pix)')
    for c_i in range(ncols):
        pl[-1,c_i].set_xlabel('dither offset (15um pix)')


    for w_i, w in enumerate(sorted(df.wavelength.unique())[::-1]):
        for r_i, r in enumerate(sorted(df.row.unique())[::-1]):
            p = pl[r_i][w_i]
            frows = df.loc[(df.wavelength == w) & (df.row == r) & (df.focus == focus)]
            if len(frows) == 0:
                continue
            if not perSpot:
                minx = frows.xpix.values[0]
                miny = frows.ypix.values[0]

            for vis in groupAllDithers(frows).visit:
                oneDither = ditherFromVisit(frows, vis)
                if perSpot:
                    minx = oneDither.xpix.values[0]
                    miny = oneDither.ypix.values[0]
                print(f'{w} {r} {vis}: {len(oneDither)} ({minx},{miny})')
                p.plot(oneDither.xpix - minx, oneDither.ypix - miny, 'x', alpha=0.5)
            p.hlines([0.0, 0.33, 0.66], -0.1, 1.2, alpha=0.3, color='k')
            p.vlines([0.0, 0.33, 0.66], -0.1, 1.2, alpha=0.3, color='k')
            if perSpot:
                p.set_ylim(-0.1, 1.1)
                p.set_xlim(-0.1, 1.1)

    finalTitle=f'dithers {df.visit.min()}..{df.visit.max()}'
    if title is not None:
        finalTitle = f'{finalTitle} {title}'
    f.suptitle(finalTitle)
    f.tight_layout()

    return f
