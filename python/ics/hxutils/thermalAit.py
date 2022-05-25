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

logger = logging.getLogger('thermalAit')
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
    # butler.addMaps(dataMapDict=butlerMaps.dataMap)

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

def plotSingleTest(plot, data, prop, testName=None,
                   label=None, color=None, useTime=False):
    testData = data if testName is None else data.loc[data.testName == testName]
    if label is None:
        label = testName

    if useTime:
        x = testData.ut
    else:
        x = testData.index
    label = f'{label} : {testData[prop].mean():0.4f}'
    ax, = plot.plot(x, testData[prop], '+-', color=color,
                    label=label)
    plot.grid(axis='both', which='both', alpha=0.5)
    plot.tick_params(axis='both', labelsize=7)

    # plot.legend(bbox_to_anchor=(1.01,1.02), loc="upper left")

    return ax, label

def loadDataForVisit0(visit0, cam, ignoreDuds=True):
    dataPath = butler.getPath('thermalData', visit=visit0, cam=cam)

    df = pd.read_csv(dataPath, sep='\s+', parse_dates=['ut'])
    if ignoreDuds:
        df = df[not df.ignore].reindex()

    return df

def ingestTestData(butler, visit0, visitN=None, overwrite=False, cam='n3'):
    """Read thermal test FITS files and generate/update pandas dataframe file

    Parameters
    ----------
    butler : `pfs.utils.butler.Butler`
        The object which known how to get to the summary file
    visit0 : `int`
        The first visit in the data set. Required.
    visitN : `int`, optional
        The last visit to load, by default None
    overwrite : `bool`
        If True, blow away output file.
    cam : str, optional
        The PFS camera name, by default 'n3'

    Writes a space-delimited data frame somewhere in /data/redux/
    """
    def alreadyDone(v, data):
        for vdone in data:
            # logger.warning(f'done test: {v} {vdone} {v == vdone}')
            if v == vdone:
                return True
        return False

    dataPath = butler.getPath('thermalData', visit=visit0, cam=cam)
    logger.warning('loading from %s', dataPath)
    if overwrite and dataPath.exists():
        dataPath.unlink()
    try:
        df = pd.read_csv(dataPath) # , sep='\s+')
        doneVisits = df.visit.values
    except FileNotFoundError:
        logger.info(f'Failed to open thermalData in {dataPath}: will create new data file.')
        df = None
        doneVisits = []

    paths = pathUtils.getPathsBetweenVisits(visit0, visitN=visitN, cam=cam)
    logger.debug(f'paths: {paths}')
    logger.debug(f'done : {doneVisits}')
    newData = []
    for p in paths:
        v = visitFromPath(p)
        if alreadyDone(v, doneVisits):
            continue

        try:
            r = hxramp.HxRamp(p)
        except Exception as e:
            print(f'failed to get unloadable visit {v} for {cam}: {e}')
            continue

        try:
            hdr = r.header()
            ts = getTimestamp(hdr)
            exptime =  hdr['EXPTIME']
            testName = hdr['OBJECT']

            cds = r.cds()
            cds /= r.rampTime
            cds *= r.e_per_ADU
            cds = cds[4:4092, 4:4092]
            s = iqr(cds)
            med = np.median(cds)
            clippedMean = np.mean(scipy.stats.sigmaclip(cds, 3.0, 3.0).clipped)

            ttp = hdr['W_XTMP7']
            tfr = hdr['W_XTMP5']
            tdet = hdr['W_XTMP12']
            tasic = hdr['W_XTMP10']
            tshield2 = hdr['W_XTMP9']
            tshield1 = hdr['W_XTMP8']
            tmangin = hdr['W_XTMP2']

            tbody1 = hdr['W_TPTMP1']
            tbody2 = hdr['W_TPTMP2']
            tbody3 = hdr['W_TPTMP3']
            tbody4 = hdr['W_TPTMP4']
            troom1 = hdr['W_TPTMP5']
            troom2 = hdr['W_TPTMP6']

            newData.append([v, ts, testName,
                            exptime, np.round(med,4), np.round(clippedMean, 4), np.round(s, 4),
                            ttp, tfr, tdet, tasic,
                            tshield2, tshield1, tmangin,
                            tbody1, tbody2, tbody3, tbody4,
                            troom1, troom2, True, False])
            print(f'{p}: {v} {med:0.3f} {clippedMean:0.3f} {s:0.3f} {ttp} {tfr} {ts}')
        except Exception as e:
            print(f'ignoring unloadable visit {v} for {cam}: {e}')

    newData = sorted(newData)
    logger.warning(f'{len(newData)} rows')
    newMeta = pd.DataFrame(newData,
                           columns=('visit', 'ut', 'testName', 'exptime',
                                    'medianFlux', 'meanFlux', '_',
                                    'plateTemp', 'frontRingTemp',
                                    'detTemp', 'asicTemp',
                                    'shield2Temp', 'shield1Temp', 'manginTemp',
                                    'bodyTemp1', 'bodyTemp2', 'bodyTemp3', 'bodyTemp4',
                                    'roomTemp1', 'roomTemp2', 'useForTest', 'ignore'),
                           )

    if df is not None:
        df = pd.concat([df, newMeta], ignore_index=True)
    else:
        df = newMeta

    def fdatetime64(ts):
        return
    logger.info(f'writing {len(df)} ({len(newMeta)} new) entries to {dataPath}')
    dataPath = pathlib.Path(dataPath)
    dataPath.parent.mkdir(mode=0o2775, parents=True, exist_ok=True)
    with open(dataPath, "w") as f:
        f.write(df.to_csv(index=False)) # , formatters=dict(ut="%Y-%m-%dT%H:%M:%S.%f")))

# Yuck. For now list labels manually:
ylabels = dict(medianFlux='ramp flux, e-/s',
               meanFlux='ramp flux, e-/s',
               plateTemp='plate temp, K',
               frontRingTemp='front ring temp, K',
               detTemp='H4 temp, K',
               asicTemp='ASIC temp, K',
               shield2Temp='shield 2 temp, K',
               shield1Temp='shield 1 temp, K',
               manginTemp='Mangin temp, K',
               bodyTemp1='body temp 1, K',
               bodyTemp2='body temp 2, K',
               bodyTemp3='body temp 3, K',
               bodyTemp4='body temp 4, K',
               roomTemp1='room temp 1, K',
               roomTemp2='room temp 2, K')

def plotTestData(butler, meta, useTime=False, tests=None, plots=None,
                 fluxRange=[0.01, 0.15]):
    print('0: plotTest: %s' % (plotSingleTest))
    try:
        plt.close('plate')
    except:
        pass

    # Completely skip masked entries
    meta = meta.loc[~meta.ignore].copy().reset_index()
    atall = meta

    if plots is None:
        plots = ['medianFlux', 'detTemp', 'plateTemp',
                 'frontRingTemp', 'bodyTemp4',
                 'shield1Temp', 'shield2Temp',
                 'roomTemp1']
    else:
        plots = ['medianFlux'] + list(plots)

    if tests is None:
        tests = meta.testName.unique()

    nplots = len(plots)
    plotsDict = dict(zip(plots, range(nplots)))
    pageHeight = 2 + nplots*2
    f, plist = plt.subplots(nrows=nplots, num='plate', clear=True, sharex=True,
                            figsize=(8,pageHeight), constrained_layout=False)
    if nplots == 1:
        plist = [plist]
    logger.warning(f'pd: {plotsDict}')
    if useTime:
        allX = atall.ut
    else:
        allX = atall.index

    labels = dict()
    axes = dict()
    for p_i, pname in enumerate(plotsDict.keys()):
        p = plist[plotsDict[pname]]

        # Always give context by plotting the whole sequence in the background.
        ax, = p.plot(allX, atall[pname], '+-', color='k', alpha=0.2)

        # Then highlight all the selected values for the active tests.
        axes = dict()
        for t in tests:
            ax, label = plotSingleTest(p, meta, pname, t, useTime=useTime)
            if p_i == 0:
                axes[t] = ax
                labels[t] = label

        # Special case the range for the main plot
        if p_i == 0:
            p.set_ylim(fluxRange)

        p.set_ylabel(ylabels[pname])

    # Give *one* x label and legend.
    xlabel = 'UT' if useTime else 'visit'
    plist[-1].set_xlabel(xlabel)

    axbox = plist[0].get_position()
    print(labels.values())
    f.legend(axes.values(), labels.values(),
             loc="upper left",
             borderaxespad=0.0,    # Small spacing around legend box
             bbox_transform=f.transFigure,
             bbox_to_anchor=[axbox.x0+axbox.width*1.1, axbox.y0+axbox.height])
    # plt.subplots_adjust(right=0.85)

    r0 = hxramp.HxRamp(pathUtils.rampPath(visit=meta.visit.min()))
    f.suptitle(f'visits {meta.visit.min()} .. {meta.visit.max()}; gain={r0.e_per_ADU} e-/ADU')
    f.tight_layout()

    return f, plotsDict, meta

def ingestMain(args=None):
    if isinstance(args, str):
        import shlex
        args = shlex.split(args)

    parser = argparse.ArgumentParser()
    parser.add_argument('--visit0', type=int, required=True,
                        help='first visit in set; used to save measurements.')
    parser.add_argument('--visitN', type=int, default=None,
                        help='last visit to add to set')
    parser.add_argument('--overwrite', action='store_true',
                        help='erase output file if true')
    parser.add_argument('--name', type=str, default='thermal',
                        help='butler experiment name, default="thermal"')
    parser.add_argument('--cam', type=str, default='n3',
                        help='PFS camera name, default "n3"')

    opts = parser.parse_args(args)
    butler = getButler(experimentName=opts.name, cam=opts.cam)

    ingestTestData(butler, visit0=opts.visit0, visitN=opts.visitN,
                   overwrite=opts.overwrite)

def plotMain(args=None):
    if isinstance(args, str):
        import shlex
        args = shlex.split(args)

    parser = argparse.ArgumentParser()
    parser.add_argument('--visit0', type=int, required=True,
                        help='first visit in set; used to find measurements.')
    parser.add_argument('--name', type=str, default='thermal',
                        help='butler experiment name, default="thermal"')
    parser.add_argument('--cam', type=str, default='n3',
                        help='PFS camera name, default "n3"')
    parser.add_argument('--plotTime', action='store_true',
                        help='Plot time on X')
    parser.add_argument('--plots', type=str, nargs='+',
                        help='list of plots -- use column names'),
    parser.add_argument('--output', type=str, default=None,
                        help='filename to write to'),
    parser.add_argument('--noSave', action='store_true',
                        help='Do not save output file')

    opts = parser.parse_args(args)
    butler = getButler(experimentName=opts.name, cam=opts.cam)

    configureMatplotlib()

    dfPath = butler.getPath('thermalData', visit=opts.visit0, cam=opts.cam)
    df = pd.read_csv(dfPath, parse_dates=['ut'])

    f, pl, df2 = plotTestData(butler, df, plots=opts.plots,
                              useTime=opts.plotTime)
    if not opts.noSave:
        outfile = opts.output
        if outfile is None:
            outfile = f'{opts.name}_{df2.visit.min()}_{df2.visit.max()}_{"time" if opts.plotTime else "ramp"}.pdf'
        f.savefig(outfile)
        print(f'wrote {outfile}')

    return f, pl, df2