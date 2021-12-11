import functools
from importlib import reload
import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
import fitsio

from pfs.utils import butler as pfsButler

from . import butlerMaps
from . import darkCube
from . import hxramp
from . import pfsutils
from . import nirander

reload(pfsButler)
reload(butlerMaps)
reload(darkCube)
reload(hxramp)
reload(pfsutils)

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

def groupDithers(ditherSet):
    ditherGroups = ditherSet.groupby(['wavelength', 'row', 'focus'])
    dig = ditherGroups['visit'].aggregate(np.min).reset_index()
    ret = ditherSet[ditherSet.visit.isin(dig.visit)]

    return ret

def selectDithers(rows, *, wavelength=None, row=None, focus=None):
    dithers = groupDithers(rows)
    if wavelength is not None:
        dithers = dithers.loc[dithers.wavelength == wavelength]
    if row is not None:
        dithers = dithers.loc[dithers.row == row]
    if focus is not None:
        dithers = dithers.loc[dithers.focus == focus]

    return dithers.reset_index(drop=True)

def measureDithers(butler, rows):
    centeredDithers = []
    centeredPeaks = []
    for r_i in range(len(rows)):
        row = rows.iloc[r_i:r_i+1].copy()
        path = nirander.ditherPath(butler, row, pfsDay='*')
        dither, hdr = fitsio.read(path, header=True)

        ctr = (np.array(dither.shape) + 1) // 2
        meas = nirander.measureSet(row, center=ctr, radius=5, ims=[dither],
                                   skipDone=False)
        ctrX = int(np.round(meas.xpix.values[0]))
        ctrY = int(np.round(meas.ypix.values[0]))
        ctr = ctrX, ctrY
        dx = int(ctrX) - meas.xpix.values[0]
        dy = int(ctrY) - meas.ypix.values[0]

        centeredDither = shiftSpot(dither, dx, dy)[0]
        peaks = nirander.measureSet(meas, center=ctr, radius=5, ims=[centeredDither],
                                   skipDone=False)
        # _, peaks = nirander.getPeaks(dith, center=ctr, radius=5)
        if len(peaks) != 1:
            raise RuntimeError(f"{len(peaks)} peaks for {path}: {peaks}")
        ee1 = centeredDither[ctrY, ctrX]
        ee3 = centeredDither[ctrY-1:ctrY+2, ctrX-1:ctrX+2].sum()
        ee5 = centeredDither[ctrY-2:ctrY+3, ctrX-2:ctrX+3].sum()
        peaks['ee1'] = ee1 / peaks.flux.values[0]
        peaks['ee3'] = ee3 / peaks.flux.values[0]
        peaks['ee5'] = ee5 / peaks.flux.values[0]

        centeredDithers.append(centeredDither)
        centeredPeaks.append(peaks)

    return centeredDithers, pd.concat(centeredPeaks)