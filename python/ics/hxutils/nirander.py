from importlib import reload
import logging
import os.path
import socket
import time
import yaml

import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
import skimage.transform
import fitsio

import sep
from matplotlib.patches import Ellipse
from scipy.spatial.distance import cdist

from pfs.utils import butler as pfsButler
from pfs.utils import spectroIds as pfsSpectroIds

from . import butlerMaps
from . import darkCube
from . import hxramp
from . import pfsutils

reload(pfsButler)
reload(pfsSpectroIds)
reload(butlerMaps)
reload(darkCube)
reload(hxramp)
reload(pfsutils)

logger = logging.getLogger('nirander')
logger.setLevel(logging.INFO)

nirButler = None
def newButler(experimentName='unnamed', cam='n1'):
    """Create a butler containing extra maps/keys for cleanroom ops. """
    global nirButler

    reload(pfsButler)
    reload(butlerMaps)

    specIds = pfsSpectroIds.SpectroIds(partName=cam)
    butler = pfsButler.Butler(specIds=specIds)
                              # configRoot=butlerMaps.configKeys['nirLabConfigRoot'])

    butler.addKeys(butlerMaps.configKeys)
    butler.addMaps(butlerMaps.configMap, butlerMaps.dataMap)
    # butler.addMaps(dataMapDict=butlerMaps.dataMap)

    butler.addKeys(dict(experimentName=experimentName))

    nirButler =  butler
    return butler
newButler()

class AidenPi(object):
    def __init__(self, name, host, port=9999, logLevel=logging.INFO):
        """Command one of Aiden's pi programs. """

        self.host = host
        self.port = port
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logLevel)

    def __str__(self):
        return f"AidenPi(name={self.name}, host={self.host}, port={self.port})"

    def __repr__(self):
        return self.__str__()

    def cmd(self, cmdStr, debug=False, maxTime=5.0):
        """ Send a single motor command.

        Args
        ----
        cmdStr : str
            Either "move x y" or a raw AllMotion command. "move" commands
            are checked against internal limit, so should always be used to
            move.
        """
        if debug:
            logFunc = self.logger.warning
        else:
            logFunc = self.logger.debug

        cmdStr = cmdStr + '\n'
        replyBuffer = ""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, self.port))
            logFunc(f'send: {cmdStr.strip()}')
            sock.sendall(bytes(cmdStr, "latin-1"))

            t0 = time.time()
            while True:
                rcvd = str(sock.recv(1024), "latin-1")
                rcvd = rcvd.strip()
                replyBuffer += rcvd
                logFunc(f'rcvd: {rcvd}, reply: {replyBuffer}')
                if replyBuffer.endswith('OK') or replyBuffer.endswith('BAD'):
                    break
                t1 = time.time()
                if t1-t0 > maxTime:
                    self.logger.fatal(f"reply timeed out after {t1-t0} seconds")
                    break
                time.sleep(0.1)

        if 'BAD' in replyBuffer:
            raise RuntimeError(f"received unknown crap: {replyBuffer}")
        if not replyBuffer.endswith('OK'):
            raise RuntimeError(f"received unknown crud: {replyBuffer}")

        parts = replyBuffer.split(';')
        return parts[0]

class PlateIlluminator:
    def __init__(self, forceLedOff=True, logLevel=logging.INFO, ip=None):

        if ip is None:
            ip = 'platepi'
        self.dev =  AidenPi('plate', ip, logLevel=logLevel)

        self.logger = logging.getLogger('plate')
        self.logger.setLevel(logLevel)
        self._led = None
        self._ledPower = None
        self._ledChangeTime = None

        self.leds = pd.DataFrame(dict(wave=['1070-0.75','1070-1','1070-1.5','1070-2','1070-2.7','1070-4'],
                                      dutyCycle=[33, 33, 33, 33, 33, 33]))
        self.leds = self.leds.set_index('wave', drop=False)

        if forceLedOff:
            self.ledsOff()

    def __str__(self):
        return f"PlateIlluminator(led={self._led}@{self._ledPower})"

    @property
    def dutyCycle(self):
        return self._ledPower

    def ledState(self):
        if self._ledChangeTime is None:
            dt = None
        else:
            dt = time.time() - self._ledChangeTime
        return (self._led, self._ledPower, dt)

    def ledOffTime(self):
        _led, _ledPower, dt = self.ledState()

        if _ledPower != 0:
            return 0
        else:
            return dt

    def ledsOff(self, debug=False):
        for w in self.leds.wave:
            self.dev.cmd(f'led {w} 0', debug=debug)
        self._led = 0
        self._ledPower = 0
        self._ledChangeTime = time.time()

    def led(self, wavelength, dutyCycle=None):
        # wavelength = int(wavelength)
        if wavelength not in self.leds.wave.values:
            raise ValueError(f"wavelength ({wavelength}) not in {self.leds.wave.to_list()}")
        if dutyCycle is None:
            dutyCycle = self.leds.dutyCycle[wavelength]

        if dutyCycle < 0 or dutyCycle > 100:
            raise ValueError(f"dutyCycle ({dutyCycle}) not in 0..100")
        dutyCycle = int(dutyCycle)

        if self._led is None:
            raise RuntimeError("current state of LEDs is unknown: need to call .ledsOff() before turning a LED on.")
        if self._led in self.leds.wave and self._led != wavelength:
            self.dev.cmd(f'led {self._led} 0', debug=True)
            self._led = 0
        self._led = wavelength
        self._ledPower = dutyCycle
        self._ledChangeTime = time.time()

        self.dev.cmd(f'led {self._led} {dutyCycle}', debug=True)

class Illuminator:
    def __init__(self, lampType='led'):
        self._wave = None
        self._power = None
        self._changeTime = None
        self._lampType = lampType

    def __str__(self):
        return f"Illuminator(type={self.lampType}: led={self._wave}@{self._power})"

    @property
    def dutyCycle(self):
        return self._power

    def ledState(self):
        if self._changeTime is None:
            dt = None
        else:
            dt = time.time() - self._changeTime
        return (self._wave, self._power, dt)

    def ledOffTime(self):
        _led, _ledPower, dt = self.ledState()

        if _ledPower != 0:
            return 0
        else:
            return dt

    def ledsOff(self, debug=False):
        for w in self.lamps.wave:
            self.dev.cmd(f'{self.lampType} {w} 0', debug=debug)
        self._wave = 0
        self._power = 0
        self._changeTime = time.time()

    def led(self, wavelength, dutyCycle=None):
        # wavelength = int(wavelength)
        if self.lampType != 'mono':
            if wavelength not in self.lamps.wave.values:
                raise ValueError(f"wavelength ({wavelength}) not in {self.lamps.wave.to_list()}")
        if dutyCycle is None:
            dutyCycle = self.leds.dutyCycle[wavelength]

        if dutyCycle < 0 or dutyCycle > 100:
            raise ValueError(f"dutyCycle ({dutyCycle}) not in 0..100")
        dutyCycle = int(dutyCycle)

        if self._wave is None:
            raise RuntimeError("current state of LEDs is unknown: need to call .ledsOff() before turning a LED on.")
        if self._wave in self.lamps.wave and self._wave != wavelength:
            self.dev.cmd(f'{self.lampType} {self._wave} 0', debug=True)
            self._wave = 0
        self._wave = wavelength
        self._power = dutyCycle
        self._changeTime = time.time()

        self.dev.cmd(f'led {self._wave} {dutyCycle}', debug=True)

def getConfig(name):
    """Load a YAML configuration file.

    This should be in pfs_instdata or ics_utils -- CPL
    """

    with open(os.path.join(os.path.expandvars('$PFS_INSTDATA_DIR'),
                           f'{name}.yaml'), 'rt') as cfgFile:
        config = yaml.safe_load(cfgFile)
    return config

class GimbalIlluminator(Illuminator):
    def __init__(self, cam='n1', forceLedOff=True, logLevel=logging.INFO, ip=None,
                 useMono=False):


        self.cam = cam

        self.logger = logging.getLogger('meade')
        self.logger.setLevel(logLevel)

        self._loadConfig()
        if useMono:
            self.lamps = self.mono
        else:
            self.lamps = self._leds
        Illuminator.__init__(self, lampType='mono' if useMono else 'led')

        if ip is None:
            ip = 'gimbalpi'
        self.dev =  AidenPi('gimbal', ip, logLevel=logLevel)

        self.preloadDistance = 50

        if forceLedOff:
            self.ledsOff()

    def __str__(self):
        return f"Meade(led={self._wave}@{self._power}, steps={self.getSteps()}, pix={self.getPix()})"

    def _loadConfig(self, cfg):
        cfg = getConfig('JHU/nirCleanroom')

        matrix = cfg['geometry']['transformCoeffs']
        self.stepToPix = skimage.transform.ProjectiveTransform(matrix=matrix)
        self.pixToStep = self.stepToPix.inverse

        self._leds = pd.DataFrame(cfg['leds'])
        self._leds = self._leds.set_index('wave', drop=False)

        # We get the monochrometer positions as mm from center of detector. Convert to pixels.
        self.mono = pd.DataFrame(cfg['mono'])
        self.mono['position'] = self.mono.positionMM / 0.015 + 2048
        self.mono = self.mono.set_index('wave', drop=False)

        self.nudges = cfg['nudges'][self.cam]


    @classmethod
    def fitTransform(cls, scans):
        t =  skimage.transform.ProjectiveTransform()

        src = scans[['xstep', 'ystep']].values.astype('f4')
        dst = scans[['xpix', 'ypix']].values.astype('f4')
        t.estimate(src, dst)

        return t

    def setTransform(self, transform):
        matrix =  transform.params
        self.stepToPix = skimage.transform.ProjectiveTransform(matrix=matrix)
        self.pixToStep = self.stepToPix.inverse

    def ledPosition(self, y, led=None):
        """Return the column for this wavelength. Currently only supports a table of wavelengths. """

        if led is None:
            led = self._wave

        return self.lamps.position[led]

    def ledFocusOffset(self, y, led=None):
        # Ignores Y, which is wrong -- CPL

        if led is None:
            led = self._wave

        return self.lamps.loc[led]['focusOffset']

    def stepsToPix(self, steps):
        steps = np.array(steps)
        upDim = steps.ndim < 2
        if upDim:
            steps = np.atleast_2d(steps)
        pix = self.stepToPix(steps)
        return pix[0] if upDim else pix

    def pixToSteps(self, pix):
        pix = np.array(pix)
        upDim = pix.ndim < 2
        if upDim:
            pix = np.atleast_2d(pix)
        steps = np.round(self.pixToStep(pix)).astype('i4')
        return steps[0] if upDim else steps
    def getSteps(self):
        xPos = int(self.dev.cmd('/1?0'))
        yPos = int(self.dev.cmd('/2?0'))

        return xPos, yPos

    def getPix(self):
        xSteps, ySteps = self.getSteps()
        return self.stepsToPix([xSteps, ySteps])

    def moveSteps(self, dx, dy):
        xPos, yPos = self.getSteps()
        self.moveToSteps(xPos+dx, yPos+dy)

    def moveToSteps(self, x, y, preload=True):
        x = int(x)
        y = int(y)
        xPos, yPos = self.getSteps()
        if preload:
            cmdStr = f"move {x-self.preloadDistance} {y-self.preloadDistance}"
            self.dev.cmd(cmdStr, debug=True, maxTime=45)

        dist = max(abs(x-xPos), abs(y-yPos))
        cmdStr = f"move {x} {y}"
        self.dev.cmd(cmdStr, debug=True, maxTime=dist/1000)

        xNew, yNew = self.getSteps()

        if x != xNew or y != yNew:
            raise RuntimeError(f'did not move right: target={x},{y}, at={xNew},{yNew}')

        return xNew, yNew

    def moveToPix(self, xpix, ypix, preload=True):
        xstep, ystep = self.pixToSteps([xpix, ypix])

        xNew, yNew = self.moveToSteps(xstep, ystep, preload=preload)
        return self.stepsToPix([xNew, yNew])

    def getTargetPosition(self, wave, row):
        """Get the final x,y pixel position for a wave+row. Applies a .nudge if one exists. """

        led = self.lamps[self.lamps.wave == wave]
        col = int(led.position)
        pos = (col, row)
        try:
            pos = self.nudges[pos]
            self.logger.info(f'nudge for ({wave}, {row}) to {pos}')
        except KeyError:
            pass

        return pos

    def home(self, doX=True, doY=True):
        """Home one or more axes. Both by default. The controller leaves it """

        if doX:
            self.dev.cmd("home x", maxTime=100)
        if doY:
            self.dev.cmd("home y", maxTime=100)
        return self.getSteps()

def takeSuperDark(meade, nexp=3, nread=3, force=False, cam='n1'):
    offtimeRequired = 3600
    offTime = meade.ledOffTime()
    if offTime < offtimeRequired:
        if not force:
            raise RuntimeError(f"need lamps to be off for {offtimeRequired}, not {offTime} seconds")

    visits = []
    for i in range(nexp):
        visit = takeRamp(cam, nread=nread)
        visits.append(visit)

    superDark = darkCube.DarkCube.createFromVisits(visits)
    return superDark

def takeRamp(cam, nread, nreset=1, exptype="test", comment="no_comment", quiet=True):
    pfsutils.oneCmd(f'hx_{cam}', f'ramp nread={nread} exptype={exptype} objname=\"{comment}\"',
                    quiet=quiet)
    visit = hxramp.pathToVisit(hxramp.lastRamp(cam=cam))

    return visit

def moveFocus(cam, piston):
    """Move the FPA focus."""

    pfsutils.oneCmd(f'xcu_{cam}', f'motors move piston={piston} abs microns')

def motorScan(meade, xpos, ypos, led=None, call=None, nread=3, posInPixels=True):
    """Move to the given positions and acquire spots.

    Args
    ----
    xpos, ypos : number or array
      The positions to acquire at. Iterates over x first.
      If one position is a scalar, is expanded to match the other axis.
      If xpos is None, use the nominal led position.
    led : (wavelength, dutyCycle)
      What lamp to turn on. If passed in, lamp is turned off at end of scan.
    nread : int
      How many reads to take per ramp
    posInPixels : bool
      Whether `xpos` and `ypos` are in pixels or steps.

    ALWAYS sorts the positions and moves below the first position
    before acquiring data.
    """

    if led is not None and led != (None,None):
        if np.isscalar(led):
            wavelength = led
            dutyCycle = None
        else:
            wavelength, dutyCycle = led
        meade.led(wavelength=wavelength, dutyCycle=dutyCycle)
        if xpos is None:
            xpos = meade.lamps.position[wavelength]

    if np.isscalar(xpos):
        if np.isscalar(ypos):
            xpos = [xpos]
            ypos = [ypos]
        else:
            xpos = [xpos]
    elif np.isscalar(ypos):
        ypos = [ypos]

    xpos = sorted(xpos)
    ypos = sorted(ypos)

    callRet = []
    lastXStep = lastYStep = 99999
    for x_i, x in enumerate(xpos):
        for y_i, y in enumerate(ypos):
            if posInPixels:
                xStep, yStep = meade.pixToSteps([x, y])
            else:
                xStep, yStep = x, y

            # We want to always move in the same direction: from low.
            preload = (xStep < lastXStep or yStep < lastYStep)
            meade.moveToSteps(xStep, yStep, preload=preload)
            lastXStep = xStep
            lastYStep = yStep

            if call is not None:
                ret = call(meade)
            else:
                ret = takeBareSpot(meade, nread=nread)
            callRet.append(ret)

    if led is not None:
        meade.ledsOff()

    if call is None:
        return pd.concat(callRet, ignore_index=True)
    else:
        return callRet

def ditherTest(meade, hxCalib, nreps=3, start=(2000,2000), npos=10):
    xrepeats = []
    yrepeats = []

    x0, y0 = meade.pixToSteps(*start)
    x0 &= ~1
    y0 &= ~1

    xscale = 5 # 5 steps = ~0.3pix
    yscale = 2 # 2 steps = ~0.3pix
    xx = np.arange(x0,x0+xscale*npos,xscale)
    yy = np.arange(y0,y0+yscale*npos,yscale)

    for i in range(nreps):
        xscan = motorScan(meade, x0, yy, led=(1085,30), posInPixels=False)
        xrepeats.append(xscan)

        yscan = motorScan(meade, xx, y0, led=(1085,30), posInPixels=False)
        yrepeats.append(yscan)

    xreps = pd.concat(xrepeats, ignore_index=True)
    measureSet(xreps, hxCalib)

    yreps = pd.concat(yrepeats, ignore_index=True)
    measureSet(yreps, hxCalib)

    return xreps, yreps

def createDither(frames, hxCalib, rad=15, doNorm=False, r1=-1):
    scale = 3

    frames = frames.sort_values('visit', ascending=True)
    ctrIdx = (scale*scale)//2
    xsteps = frames['xstep'].unique()
    ysteps = frames['ystep'].unique()

    xoffsets = {xs:(scale-1)-xi for xi,xs in enumerate(xsteps)}
    yoffsets = {ys:(scale-1)-yi for yi,ys in enumerate(ysteps)}
    # Need better sanity checks
    if len(frames) != scale*scale or len(xsteps) != scale or len(ysteps) != scale:
        raise ValueError("only want to deal with 3x3 dithers")

    ctr = np.round(frames[['xpix','ypix']].values[ctrIdx]).astype('i4')
    xslice = slice(ctr[0]-rad, ctr[0]+rad)
    yslice = slice(ctr[1]-rad, ctr[1]+rad)

    outIm = np.zeros((rad*2*scale, rad*2*scale), dtype='f4')
    bkgndMask = np.ones((rad*2, rad*2), dtype='f4')
    bkgndMask[2:-2, 2:-1] = 0
    maskIm = hxCalib.badMask[yslice,xslice]
    bkgndMask *= 1-(maskIm>0)
    print(f"{bkgndMask.sum()}/{bkgndMask.size}")
    inIms = []
    outIms = []
    for f_i, fIdx in enumerate(frames.index):
        f1 = frames.loc[fIdx]
        im = hxCalib.isr(int(f1.visit), r1=r1)
        im = im[yslice,xslice].astype('f4')
        maskedIm = im*bkgndMask
        bkgnd = np.median(maskedIm[np.where(maskedIm > 0)])
        im -= bkgnd
        maskedIm = im*bkgndMask

        if f_i == 0:
            normSum = np.sum(maskedIm, dtype='f8')
        imSum = np.sum(maskedIm, dtype='f8')
        if doNorm:
            im *= (normSum/imSum).astype('f4')
        xoff = xoffsets[f1.xstep]
        yoff = yoffsets[f1.ystep]
        print(f'{f1.visit:0.0f}: wave: {f1.wavelength} focus: {f1.focus} '
              f'pix: {xoff} {yoff} step: {f1.xstep:0.0f},{f1.ystep:0.0f} '
              f'ctr: {f1.xpix:0.2f},{f1.ypix:0.2f} bkgnd: {bkgnd:0.3f} '
              f'scale: {normSum:0.1f}/{imSum:0.1f}={normSum/imSum:0.3f}')
        outIm[yoff::scale, xoff::scale] = im
        inIms.append(im)

        out1 = outIm*0
        out1[yoff::scale, xoff::scale] = im
        outIms.append(out1)

    return outIm, inIms, outIms

def ditherPath(butler, row, pfsDay=None):
    if pfsDay is None:
        path = butler.get('dither',
                          idDict=dict(visit=int(row.visit),
                                      wave=int(row.wavelength),
                                      focus=row.focus,
                                      row=(np.round(row.ypix/100)*100)))
    else:
        path = butler.search('dither',
                             idDict=dict(visit=int(row.visit),
                                         wave=int(row.wavelength),
                                         focus=row.focus,
                                         row=(np.round(row.ypix/100)*100)),
                             pfsDay=pfsDay)

    return path

def ditherPaths(butler, rows, pfsDay='*'):
    paths = []
    for i in range(0, len(rows), 9):
        row = rows.iloc[i]
        path = ditherPath(butler, row, pfsDay=pfsDay)[0]
        paths.append(path)
    return paths

def allDithers(frames, hxCalib, rad=15, butler=None, doNorm=False):
    dithers = []
    ids = []
    for i in range(len(frames)//9):
        dithFrames = frames.iloc[i*9:(i+1)*9]
        print(len(dithFrames))
        dith1, _, _ = createDither(dithFrames, hxCalib, rad=rad, doNorm=doNorm)
        dithers.append(dith1)

        if butler is not None:
            row = dithFrames.iloc[0]
            idDict = dict(visit=int(row.visit),
                          wave=int(row.wavelength),
                          focus=row.focus,
                          row=(np.round(row.ypix/100)*100))
            ids.append(idDict)
            path = butler.get('dither', idDict=idDict)
            hdr = [dict(name='VISIT', value=int(row.visit), comment="visit of 0,0 image"),
                   dict(name='WAVE', value=row.wavelength),
                   dict(name='FOCUS', value=row.focus),
                   dict(name='XPIX', value=row.xpix, comment="measured xc of 0,0 image"),
                   dict(name='YPIX', value=row.ypix, comment="measured yc of 0,0 image"),
                   dict(name='XSTEP', value=int(row.xstep)),
                   dict(name='YSTEP', value=int(row.ystep)),
                   dict(name='SIZE', value=row.size, comment="measured RMS of 0,0 image"),
                   dict(name='FLUX', value=row.flux, comment="measured total flux of 0,0 image"),
                   dict(name='PEAK', value=row.peak, comment="measured peak of 0,0 image")]
            path.parent.mkdir(parents=True, exist_ok=True)
            fitsio.write(path, dith1, header=hdr, clobber=True)
            logger.info(f'wrote dither {path}')
    return ids

def ditherAt(meade, led, row, nramps=3, npos=3, nread=3, xsteps=5, ysteps=2):
    """Acquire dithered imaged at a given position. """

    if npos%2 != 1:
        raise ValueError("not willing to deal with non-odd dithering")
    rad = npos//2
    xc, yc = meade.pixToSteps([meade.leds.position[led], row])
    x0, y0 = xc-(rad*xsteps), yc-(rad*ysteps)

    xx = x0 + np.arange(npos)*xsteps
    yy = y0 + np.arange(npos)*ysteps

    visits = []
    for r_i in range(nramps):
        gridVisits = motorScan(meade, xx, yy, led=led, posInPixels=False)
        visits.extend(gridVisits)

    return pd.concat(visits, ignore_index=True)

def ditherAtPix(meade, pos, npos=3, nread=3, xsteps=5, ysteps=2):
    """Acquire set of dithered starting from the given pixel position. """

    if npos%2 != 1:
        raise ValueError("not willing to deal with non-odd dithering")
    rad = npos//2
    xc, yc = meade.pixToSteps(pos)
    x0, y0 = xc-(rad*xsteps), yc-(rad*ysteps)

    xx = x0 + np.arange(npos)*xsteps
    yy = y0 + np.arange(npos)*ysteps

    ditherVisits = motorScan(meade, xx, yy, nread=nread, posInPixels=False)

    return ditherVisits

def ditherSet(meade, butler=None, waves=None, rows=[88,2040,3995], focus=122.0,
              nramps=1, takeDarks=False):
    if waves is None:
        waves = meade.leds.wave
    if np.isscalar(waves):
        waves = [waves]

    if np.isscalar(rows):
        rows = [rows]
    rows = np.array(rows, dtype='f4')

    if np.isscalar(focus):
        focus = [focus]
    focus = np.array(focus, dtype='f4')

    ditherList = []
    try:
        for w_i, w in enumerate(waves):
            for r_i, row in enumerate(rows):
                if takeDarks:
                    meade.ledsOff()
                    xc, yc = meade.pixToSteps([meade.leds.position[w], row])
                    meade.moveToSteps(xc, yc, preload=True)
                    # Take and save a fresh dark
                    dark = takeSuperDark(meade, force=True, nread=3, nexp=5)
                    nirButler.put(dark, 'dark', dict(visit=dark.visit0))
                meade.led(w)
                led, dutyCycle, _ = meade.ledState()
                for f_i, f in enumerate(focus):
                    print(f"led {w} on row {row} with focus {f}")
                    pfsutils.oneCmd('xcu_n1', f'motors move piston={f} abs microns')
                    try:
                        ret = ditherAt(meade, w, row, nramps=nramps)
                    except Exception as e:
                        raise

                    ret['focus'] = f
                    ret['wavelength'] = w
                    ret['dutyCycle'] = dutyCycle
                    ditherList.append(ret)

                    print("ditherList: ", len(ditherList))
                    rowFrame =  pd.concat(ditherList, ignore_index=True)
                    if butler is not None:
                        outFileName = writeMeasures(butler, rowFrame)
                        print(f"wrote {len(rowFrame)} lines to {outFileName} at led {w} on row {row} with focus {f}")
    except Exception as e:
        print(f'oops: {e}')
        # breakpoint()
        raise
    finally:
        meade.ledsOff()
        return ditherList

def spotSet(meade, butler=None, waves=None, rows=None, focus=None,
            doDither=False, nread=3):
    """Primary acquisition routine."""

    if waves is None:
        waves = meade.leds.wave
    if np.isscalar(waves):
        waves = [waves]

    if np.isscalar(rows):
        rows = [rows]
    rows = np.array(rows, dtype='f4')

    if np.isscalar(focus):
        focus = [focus]
    focus = np.array(focus, dtype='f4')

    spotList = []
    try:
        for w_i, w in enumerate(waves):
            meade.led(w)
            led, dutyCycle, _ = meade.ledState()
            for r_i, row in enumerate(rows):
                pos = meade.getTargetPosition(w, row)
                if not doDither:
                    meade.moveToPix(*pos, preload=True)
                for f_i, f in enumerate(focus):
                    print(f"led {w} on row {row} with focus {f}")
                    pfsutils.oneCmd('xcu_n1', f'motors move piston={f} abs microns')
                    try:
                        if doDither:
                            meas = ditherAtPix(meade, pos=pos, nread=3)
                        else:
                            meas = takeBareSpot(meade, nread=3,
                                                comment=f'spotSet_{w}_{round(row)}_{round(f)}')
                    except Exception as e:
                        raise

                    meas['focus'] = f
                    meas['wavelength'] = w
                    meas['dutyCycle'] = dutyCycle
                    spotList.append(meas)

                    rowFrame = pd.concat(spotList, ignore_index=True)
                    if butler is not None:
                        outFileName = writeMeasures(butler, rowFrame)
                        print(f"wrote {len(rowFrame)} lines to {outFileName} "
                              f"at led {w} on row {row} with focus {f}")
    except Exception as e:
        print(f'oops: {e}')
        # breakpoint()
        raise
    finally:
        meade.ledsOff()

    return pd.concat(spotList, ignore_index=True)

def writeMeasures(butler, df):
    outFileName = butler.getPath('measures', idDict=dict(visit=df.visit.min()))
    outFileName.parent.mkdir(mode=0o2775, parents=True, exist_ok=True)
    with open(outFileName, mode='w') as outf:
        outf.write(df.to_string())

    return outFileName

def trimRect(im, c, r=100):
    cx, cy = c
    im2 = im[cy-r:cy+r, cx-r:cx+r]

    return im2.copy()

def getPeaks(im, thresh=250.0, mask=None, center=None, radius=100,
             convolveSigma=None, kernel=None):
    bkgnd = sep.Background(im, mask=mask)
    bkg = np.array(bkgnd)
    corrImg = im - bkg

    if convolveSigma is not None:
        corrImg = scipy.ndimage.gaussian_filter(corrImg, sigma=convolveSigma)

    # Doing this the expensive way: extract on full image, then trim
    spots = sep.extract(corrImg, deblend_cont=1.0,
                        thresh=thresh, mask=mask,
                        filter_kernel=kernel)
    spotsFrame = pd.DataFrame(spots)
    spotsFrame['ellipticity'] = spotsFrame.x2 / spotsFrame.y2
    spotsFrame.loc[spotsFrame.ellipticity < 1,
                   'ellipticity'] = 1/spotsFrame.loc[spotsFrame.ellipticity < 1,
                                                     'ellipticity']
    if center is not None:
        center = np.atleast_2d(center)
        keep_w = cdist(spotsFrame[["x","y"]], center) <= radius
        spotsFrame = spotsFrame.loc[keep_w]

    return corrImg, spotsFrame

def showPeaks(corrImg, spots, d, mask=None):
    d.set('frame delete all')
    d.set('frame new')
    d.set_np2arr(corrImg)

    if mask is not None:
        d.set(f'array mask [xdim={mask.shape[1]},ydim={mask.shape[0]},bitpix="16"]', mask)
        d.set('mask transparency 75')

    if spots is not None:
        spotSpecs = ['image']
        for s in spots:
            spotSpecs.append(r'ellipse {s["x"]} {s["y"]} {s["a"]} {s["b"]} {s["theta"] * 180/np.pi}')

        d.set('regions', ';'.join(spotSpecs))

def showPeaks2(corrImg, spots=None, nsig=3.0, imrad=100):
    f, pl = plt.subplots(figsize=(8,8))
    p1 = pl

    mn = np.median(corrImg)
    sd = np.std(corrImg)
    if np.isscalar(nsig):
        lowSig = highSig = nsig, nsig
    else:
        lowSig, highSig = nsig
    p1map = p1.imshow(corrImg, vmin=mn-sd*lowSig, vmax=mn+sd*highSig,
                      aspect='equal')
    f.colorbar(p1map)

    if spots is not None:
        for s in spots:

            # plot an ellipse for each object
            e = Ellipse(xy=(s['x'], s['y']),
                        width=6*s['a'],
                        height=6*s['b'],
                        angle=s['theta'] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            p1.add_artist(e)

        if len(spots) == 1:
            x, y = spots[0]['x'], spots[0]['y']

            p1.set_xlim(x-imrad, x+imrad)
            p1.set_ylim(y-imrad, y+imrad)

    return f

def bestSpot(spots, ctr):
    dist = np.sqrt((spots['x']-ctr[0])**2 + (spots['y']-ctr[1])**2)
    closest = np.argmin(dist)
    return spots[closest:closest+1]

def brightestSpot(spots):
    order = np.argsort(spots['flux'])[::-1]
    i=0
    while i < len(order):
        if spots['flag'][i] != 0:
            print('skipping')
            i += 1
            continue
        return spots[i:i+1]
    return None

def getBestFocus(sweep):
    sweep = sweep.dropna()
    x = sweep.sort_values(by='focus')['focus']
    y = sweep.sort_values(by='focus')['size']

    poly = np.polynomial.Polynomial.fit(x,y,2)
    c,b,a = poly.convert().coef
    minx = -b/(2*a)
    return minx, poly

def getFocusGrid(center, spacing=2, r=5):
    focusReq = center + (np.arange(2*r-1) - (r-1))*spacing
    return focusReq

def _scanForFocus(center, spacing, r, nread=3, cam='n1', measureCall=None):
    focusReq = getFocusGrid(center, spacing=spacing, r=r)
    print(focusReq)

    if focusReq[0] < 15:
        raise RuntimeError(f"focusReq[0] too low, not starting below: focusReq")

    pfsutils.oneCmd(f'xcu_{cam}', f"motors move piston={focusReq[0]-10} abs microns")

    visits = []
    for f in focusReq:
        pfsutils.oneCmd(f'xcu_{cam}', f"motors move piston={f} abs microns")
        visit = takeRamp(cam=cam, nread=nread, comment=f'focus_sweep:{f}')
        visits.append(visit)

    scanFrame = pd.DataFrame(dict(visit=visits, focus=focusReq))

    if measureCall is not None:
        try:
            focusSet = measureCall(scanFrame)
            bestFocus, focusPoly = getBestFocus(focusSet)
            print(f"best focus: {bestFocus}")
            if (bestFocus is not None and
                bestFocus >= focusReq[0] and
                bestFocus <= focusReq[-1]):

                pfsutils.oneCmd(f'xcu_{cam}', f"motors move piston={bestFocus} abs microns")
                visit = takeRamp(cam=cam, nread=nread, comment=f'at_best_focus:{bestFocus}')
                visits.append(visit)

                bestSize = focusPoly(bestFocus)
                bestFrame = pd.DataFrame(dict(visit=[visit], focus=[bestFocus]))
                measureCall(bestFrame)
                print(f"expected {bestSize:0.2f}, got {bestFrame['size'].values[0]:0.2f} ")

                scanFrame = pd.concat([scanFrame, bestFrame], ignore_index=True)
        except Exception as e:
            print(f"Failed to measure and go to best focus: {e}")

    return scanFrame

def basicDataFrame(meade, visits, focus=None):
    """Create the core dataframe for some visits. Queries the controller for step/led info"""

    if np.isscalar(visits):
        visits = [visits]
    scanFrame = pd.DataFrame(dict(visit=visits))
    if focus is not None:
        scanFrame['focus'] = focus

    wavelength, dutyCycle, _ = meade.ledState()
    xstep, ystep = meade.getSteps()

    scanFrame['wavelength'] = wavelength
    scanFrame['xstep'] = xstep
    scanFrame['ystep'] = ystep
    scanFrame['dutyCycle'] = dutyCycle

    return scanFrame

def focusSweep(meade, led=None, pix=None, centerFocus=None, spacing=10, r=5, measureCall=None):
    """Setup the gimbal to the given pixel and led, and make a focus sweep """

    doLedOff = led is not None

    if led is not None:
        meade.led(led)
    wavelength, dutyCycle, _ = meade.ledState()

    if pix is not None:
        meade.moveToPix(*pix)
    xpix, ypix = meade.getPix()
    xstep, ystep = meade.getSteps()

    focusScan = scanForFocus(centerFocus, spacing=spacing, r=r,
                             measureCall=measureCall)

    if doLedOff:
        meade.ledsOff()

    focusScan['wavelength'] = wavelength
    focusScan['xstep'] = xstep
    focusScan['ystep'] = ystep
    focusScan['dutyCycle'] = dutyCycle

    return focusScan

def scanForFocus(center, spacing=5, r=4, measureCall=None):
    return _scanForFocus(center, spacing=spacing, r=r, measureCall=measureCall)
def scanForCrudeFocus(center, spacing=25, r=3, measureCall=None):
    return _scanForFocus(center, spacing=spacing, r=r, measureCall=measureCall)

def measureSet(scans, meade, hxCalib=None, thresh=1000, center=None,
               radius=100, skipDone=True, ims=None, trimBad=True,
               convolveSigma=None, kernel=None, r0=0, r1=-1):
    """Measure the best spots in a DataFrame of images

    Parameters
    ----------
    scans : `pd.DataFrame`
        [description]
    hxCalib : `HxCalib`
        An object which can .isr() an image.
    thresh : int, optional
        The detection threshold in ADU, by default 250
    center : (float, float), optional
        The expected pixel center, by default None
    radius : int, optional
        How far to look for a spot, by default 100
    skipDone : bool, optional
        Do not reprocess already measured rows, by default True

    Returns
    -------
    [type]
        [description]
    """

    if hxCalib is None:
        hxCalib = hxramp.HxCalib()

    for f in 'x2', 'y2', 'xpix', 'ypix', 'flux', 'peak', 'size':
        if f not in scans:
            scans[f] = np.nan

    if skipDone:
        notDone = scans[scans.xpix.isna()].index
    else:
        notDone = scans.index

    for scan_i in notDone:
        if center is False:
            center_i = None
        elif center is None:
            # Try for a measured position. Failing that, use the steps.
            center_i = (scans.loc[scan_i, 'xpix'], scans.loc[scan_i, 'ypix'])
            if np.isnan(center_i[0]) or np.isnan(center_i[1]):
                try:
                    stepCenter = (scans.loc[scan_i, 'xstep'], scans.loc[scan_i, 'ystep'])
                    center_i = meade.stepsToPix(stepCenter)
                    logger.info((f"{scan_i} center from steps: {center_i}"))
                except Exception as e:
                    logger.warn(f'failed to get a center for {scans.loc[scan_i]}: {e}')
                    center_i = None
            else:
                logger.info((f"{scan_i} center from pix: {center_i}"))

        else:
            center_i = center

        if ims is not None:
            corrImg = ims[scan_i]
            center_i = None
        else:
            if hxCalib is not None:
                corrImg = hxCalib.isr(scans.loc[scan_i, 'visit'], r0=r0, r1=r1)
            else:
                ramp = hxramp.HxRamp(visit=scans.loc[scan_i, 'visit'])
                corrImg = ramp.cdsN(r0=r0, r1=r1)

        corrImg, spots = getPeaks(corrImg,
                                  center=center_i, radius=radius,
                                  thresh=thresh,
                                  mask=hxCalib.badMask,
                                  convolveSigma=convolveSigma,
                                  kernel=kernel)
        if trimBad and len(spots) > 0:
            ok = (spots.size < 150) & (spots.ellipticity < 5.0)
            origSpots = spots.copy()
            spots = spots[ok]
        if spots is None or len(spots) == 0:
            print(f"nope: i={scan_i}, scan={scans.loc[scan_i]}")
        else:
            print(f"    : i={scan_i}, visit={scans.loc[scan_i, 'visit']}, nspots={len(spots)}")
            bestSpot = spots.loc[spots.flux.idxmax()]
            scans.loc[scan_i, 'xpix'] = bestSpot.x
            scans.loc[scan_i, 'ypix'] = bestSpot.y
            scans.loc[scan_i, 'x2'] = bestSpot.x2
            scans.loc[scan_i, 'y2'] = bestSpot.y2
            scans.loc[scan_i, 'size'] =  (bestSpot.x2 + bestSpot.y2)/2
            scans.loc[scan_i, 'flux'] = bestSpot.flux
            scans.loc[scan_i, 'peak'] = bestSpot.peak

    return scans

def takeBareSpot(meade, nread=3, comment=None):
    """Lowest-level exposure which returns a dataframe with (visit, xstep, ystep, led) """

    visit = takeRamp(cam=meade.cam, nread=nread, exptype='object', comment=comment)
    df = basicDataFrame(meade, visits=[visit])

    return df

def takeSpot(meade, pos=None, focus=None, light=None, nread=3, comment=None):
    """Lowest-level exposure which optionally sets focus/led/gimbal and returns a dataframe. """

    if pos is not None:
        meade.moveToPix(*pos)
    if focus is not None:
        pfsutils.oneCmd('xcu_n1', f'motors move piston={focus} abs microns')
    if light is not None:
        if np.isscalar(light):
            meade.led(light)
        else:
            meade.led(*light)
    df = takeBareSpot(meade, nread=nread, comment=comment)
    if focus is not None:
        df['focus'] = focus

    if light is not None:
        meade.ledsOff()
    return df
