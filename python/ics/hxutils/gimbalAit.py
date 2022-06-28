from importlib import reload

import argparse
import logging
import pathlib

import matplotlib.pyplot as plt

import scipy.stats
import numpy as np
import pandas as pd

from pfs.utils import butler as icsButler
from pfs.utils import spectroIds

from . import hxramp
from . import pathUtils
from . import butlerMaps
from . import nirander

logger = logging.getLogger('gimbalAit')
logger.setLevel(logging.INFO)

def configureMatplotlib():
    import matplotlib as mpl

    mpl.rcParams['lines.linewidth'] = 0.5
    mpl.rcParams['lines.markersize'] = 4.0
    mpl.rcParams['legend.fontsize'] = 'x-small'
    mpl.rcParams['axes.labelsize'] = 'small'
    mpl.rcParams['text.usetex'] = False

def getButler(experimentName, cam):
    butler = icsButler.Butler(specIds=spectroIds.SpectroIds(cam, site='J'))
    butler.addKeys(butlerMaps.configKeys)
    butler.addMaps(butlerMaps.configMap, butlerMaps.dataMap)

    butler.addKeys(dict(experimentName=experimentName))

    return butler

def visitFromPath(p):
    name = pathlib.Path(p).stem

    return int(name[4:-2], base=10)

def iqr(im):
    stats = np.percentile(im, [25, 75])
    return stats[1] - stats[0]

def getTimestamp(hdr):
    dt = np.datetime64(f'{hdr["DATE-OBS"]}T{hdr["UT-STR"]}')
    return dt

def loadDataForVisit0(visit0, cam, ignoreDuds=True):
    dataPath = butler.getPath('gimbalData', visit=visit0, cam=cam)

    df = pd.read_csv(dataPath, sep='\s+', parse_dates=['ut'])
    if ignoreDuds:
        df = df[not df.ignore].reindex()

    return df

def takeGrid(opts):
    butler = getButler(experimentName=opts.name, cam=opts.cam)

    reload(nirander)
    meade = nirander.GimbalIlluminator(cam=opts.cam,
                                       forceLedOff=True,
                                       lampType='mono')
    logger.info('meade=%s', meade)


def takeGridMain(args=None):
    if isinstance(args, str):
        import shlex
        args = shlex.split(args)

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='grid',
                        help='butler experiment name, default="grid"')
    parser.add_argument('--cam', type=str, default='n3',
                        help='PFS camera name, default "n3"')
    parser.add_argument('--waves', type=str,
                        help='comma-delimited list of known wavelengths')
    parser.add_argument('--rows', type=str,
                        help='comma-delimited list of known rows')
    parser.add_argument('--focus', type=str,
                        help='comma-delimited list of focus positions')

    opts = parser.parse_args(args)


    if opts.waves is not None:
        opts.waves = [int(w) for w in opts.waves.split(',')]
    if opts.rows is not None:
        opts.rows = [int(r) for r in opts.rows.split(',')]
    if opts.focus is not None:
        opts.focus = [float(f) for f in opts.focus.split(',')]

    print(opts)

    # takeGrid(opts)
