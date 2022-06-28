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
from . import hxcalib
from . import pathUtils
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

class AidanPi(object):
    def __init__(self, name, host, port=9999, logLevel=logging.INFO):
        """Command one of Aidan's pi programs. """

        self.host = host
        self.port = port
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logLevel)

    def __str__(self):
        return f"AidanPi(name={self.name}, host={self.host}, port={self.port})"

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
                    self.logger.fatal(f"reply timed out after {t1-t0} seconds")
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
        self.dev =  AidanPi('plate', ip, logLevel=logLevel)

        self.logger = logging.getLogger('plate')
        self.logger.setLevel(logLevel)
        self._led = None
        self._ledPower = None
        self._ledChangeTime = None

        self.lamps = pd.DataFrame(dict(wave=['1070-0.75','1070-1','1070-1.5','1070-2','1070-2.7','1070-4'],
                                      dutyCycle=[33, 33, 33, 33, 33, 33]))
        self.lamps = self.lamps.set_index('wave', drop=False)

        if forceLedOff:
            self.lampsOff()

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
        for w in self.lamps.wave:
            self.dev.cmd(f'led {w} 0', debug=debug)
        self._led = 0
        self._ledPower = 0
        self._ledChangeTime = time.time()

    def led(self, wavelength, dutyCycle=None):
        # wavelength = int(wavelength)
        if wavelength not in self.lamps.wave.values:
            raise ValueError(f"wavelength ({wavelength}) not in {self.lamps.wave.to_list()}")
        if dutyCycle is None:
            dutyCycle = self.lamps.dutyCycle[wavelength]

        if dutyCycle < 0 or dutyCycle > 100:
            raise ValueError(f"dutyCycle ({dutyCycle}) not in 0..100")
        dutyCycle = int(dutyCycle)

        if self._led is None:
            raise RuntimeError("current state of LEDs is unknown: need to call .lampsOff() before turning a LED on.")
        if self._led in self.lamps.wave and self._led != wavelength:
            self.dev.cmd(f'led {self._led} 0', debug=True)
            self._led = 0
        self._led = wavelength
        self._ledPower = dutyCycle
        self._ledChangeTime = time.time()

        self.dev.cmd(f'led {self._led} {dutyCycle}', debug=True)

class Illuminator:
    def __init__(self):
        self._wave = None
        self._power = None
        self._changeTime = None

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

    def lampsOff(self, debug=True):
        if self.lampType == 'mono':
            self.dev.cmd('mono off', debug=debug)
        else:
            for w in self.lamps.wave:
                self.dev.cmd(f'{self.lampType} {w} 0', debug=debug)
        self._wave = 0
        self._power = 0
        self._changeTime = time.time()

    def _monoOn(self, wavelength, dutyCycle=100):
        self.dev.cmd(f'mono {self._wave}', debug=True)
        self.dev.cmd(f'mono on', debug=True)

        self._wave = wavelength
        self._power = dutyCycle
        self._changeTime = time.time()

    def _ledOn(self, wavelength, dutyCycle=None):
        pass

    def led(self, wavelength, dutyCycle=None):
        # wavelength = int(wavelength)
        if self.lampType != 'mono':
            if wavelength not in self.lamps.wave.values:
                raise ValueError(f"wavelength ({wavelength}) not in {self.lamps.wave.to_list()}")
        if dutyCycle is None:
            dutyCycle = self.lamps.dutyCycle[wavelength]
        if self.lampType == 'mono':
            if dutyCycle != 100:
                raise ValueError('monochrometer duty cycle is not controllable')

        if dutyCycle < 0 or dutyCycle > 100:
            raise ValueError(f"dutyCycle ({dutyCycle}) not in 0..100")
        dutyCycle = int(dutyCycle)

        if self._wave is None:
            raise RuntimeError("current state of LEDs is unknown: need to call .lampsOff() before turning a LED on.")
        if self._wave in self.lamps.wave and self._wave != wavelength:
            if self.lampType == 'mono':
                self.dev.cmd('mono off')
            else:
                self.dev.cmd(f'{self.lampType} {self._wave} 0', debug=True)
            self._wave = 0

        self._wave = wavelength
        self._power = dutyCycle
        self._changeTime = time.time()

        if self.lampType == 'mono':
            self.dev.cmd(f'mono {self._wave}', debug=True)
            self.dev.cmd(f'mono on', debug=True)
        else:
            self.dev.cmd(f'{self.lampType} {self._wave} {dutyCycle}', debug=True)

class LedControl(Illuminator):
    pass
class Mono:
    def __init__(self):
        self._wave = 0
        self._power = 0
        self._changeTime = None

    def lampStatus(self):
        if self._changeTime is None:
            dt = None
        else:
            dt = time.time() - self._changeTime

        ret = self.dev.cmd('mono ?')
        return (self._wave, self._power, dt, ret)

def getConfig(name, subdirectory=''):
    """Load a YAML configuration file.

    This should be in pfs_instdata or ics_utils -- CPL
    """

    with open(os.path.join(os.path.expandvars('$PFS_INSTDATA_DIR'),
                           'config', subdirectory, f'{name}.yaml'), 'rt') as cfgFile:
        config = yaml.safe_load(cfgFile)
    return config

class GimbalIlluminator(Illuminator):
    knownLampTypes = {'led', 'mono'}

    def __init__(self, cam, forceLedOff=True, logLevel=logging.INFO, ip=None,
                 lampType='mono'):

        self.cam = cam

        self.logger = logging.getLogger('meade')
        self.logger.setLevel(logLevel)

        if lampType not in self.knownLampTypes:
            raise ValueError(f'unknown lamptype {lampType}: need {self.knownLampTypes}')
        self.lampType = lampType

        self._loadConfig()
        Illuminator.__init__(self)

        if ip is None:
            ip = 'gimbalpi'
        self.dev =  AidanPi('gimbal', ip, logLevel=logLevel)

        self.preloadDistance = 50

        if forceLedOff:
            self.lampsOff()

    def __str__(self):
        return f"Gimbalator(type={self.lampType}, led={self._wave}@{self._power}, steps={self.getSteps()}, pix={self.getPix()})"

    def _resolveTransform(self, transformName):
        """Given a fully resolved transform class name, get the class or detonate."""
        parts = transformName.split('.')
        ns_o = globals()
        for p in parts:
            if isinstance(ns_o, dict):
                ns_o = ns_o[p]
            else:
                ns_o = getattr(ns_o, p)
        return ns_o

    def _loadConfig(self):
        cfg = getConfig('nirCleanroom', subdirectory='JHU')

        transformClass = self._resolveTransform(cfg['geometry']['transformClass'])
        coeffs = cfg['geometry']['transformCoeffs']
        if isinstance(coeffs, (list, tuple)):
            assert len(coeffs) == 2, "lists of coeffs be len=2"

            self._stepToPix = transformClass(np.array(coeffs[0]))
            self._pixToStep = transformClass(np.array(coeffs[1]))
        else:
            self._stepToPix = transformClass(np.array(coeffs))
            self._pixToStep = self._stepToPix.inverse

        self._leds = pd.DataFrame(cfg['leds'])
        self._leds = self._leds.set_index('wave', drop=False)

        # We get the monochrometer positions as mm from center of detector. Convert to pixels.
        self.mono = pd.DataFrame(cfg['mono'])
        self.mono['position'] = self.mono.positionMM / 0.015 + 2048
        self.mono = self.mono.set_index('wave', drop=False)

        # The keys are strings containing a pair: "(940, 2040)"
        # We do not eval here but encode when indexing.
        self.nudges = cfg['nudges'][self.cam]
        self.logger.info(f'nudges: {self.nudges}')
        if self.lampType == 'mono':
            self.lamps = self.mono
        else:
            self.lamps = self._leds

    @classmethod
    def fitTransform(cls, scans, transform=None, transformArgs=None):
        if transform is None:
            transform = skimage.transform.PolynomialTransform
        t = transform()

        if transformArgs is None:
            transformArgs = dict()
            if transform == skimage.transform.PolynomialTransform:
                transformArgs['order'] = 3

        src = scans[['xstep', 'ystep']].values.astype('f4')
        dst = scans[['xpix', 'ypix']].values.astype('f4')
        t.estimate(src, dst, **transformArgs)

        # Polynomial xform has no inverse. Need to fit both ways.
        if transform != skimage.transform.PolynomialTransform:
            return t

        tinv = transform()
        tinv.estimate(dst, src, **transformArgs)

        return t, tinv

    def updateTargets(self, df):
        """Utility hack to retransform pix0 = transform(step), assuming trnsform has changed. """
        xtargets = []
        ytargets = []
        for i in range(len(df)):
            row = df.iloc[i]
            xstep, ystep = row[['xstep', 'ystep']]
            target = self.stepsToPix((xstep, ystep))
            xtargets.append(target[0])
            ytargets.append(target[1])

        df['xpix0'] = xtargets
        df['ypix0'] = ytargets

    def setTransform(self, transforms):
        try:
            t, tinv = transforms
        except:
            t = transforms
            tinv =  None

        self._stepToPix = t.__class__(t.params)
        if tinv is None:
            self._pixToStep = self._stepToPix.inverse
        else:
            self._pixToStep = tinv.__class__(tinv.params)

    def ledPosition(self, y, wave=None):
        """Return the column for this wavelength. Currently only supports a table of wavelengths. """

        if wave is None:
            wave = self._wave

        return self.lamps.position[wave]

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
        pix = self._stepToPix(steps)
        return pix[0] if upDim else pix

    def pixToSteps(self, pix):
        pix = np.array(pix)
        upDim = pix.ndim < 2
        if upDim:
            pix = np.atleast_2d(pix)
        steps = np.round(self._pixToStep(pix)).astype('i4')
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

    def moveToSteps(self, x, y, preload=True, onlyIfNecessary=True):
        """Move the gimbelator to the given (x,y) steps. Apply preload and shortcircuit by default.

        Parameters
        ----------
        x : `int`
            Steps for the X stage
        y : `int`
            Steps for the Y stage
        preload : bool, optional
            Whether to always come from the negative side, by default True
        onlyIfNecessary : bool, optional
            Whether to skip commands if we are already there, by default True

        Returns
        -------
        xpos, ypos : `tuple`
            The final position.
        """
        x = int(x)
        y = int(y)
        xPos, yPos = self.getSteps()
        if onlyIfNecessary and xPos == x and yPos == y:
            return xPos, yPos

        # For the preload, if *either* motor is > requested pos, move both below target.
        if preload and (x < xPos or y < yPos):
            cmdStr = f"move {x-self.preloadDistance} {y-self.preloadDistance}"
            self.dev.cmd(cmdStr, debug=True, maxTime=45)

        dist = max(abs(x-xPos), abs(y-yPos))
        cmdStr = f"move {x} {y}"
        self.dev.cmd(cmdStr, debug=True, maxTime=dist/1000)

        xNew, yNew = self.getSteps()

        if x != xNew or y != yNew:
            raise RuntimeError(f'did not move right: target={x},{y}, at={xNew},{yNew}')

        return xNew, yNew

    def moveToPix(self, xpix, ypix, preload=True, onlyIfNecessary=True):
        xstep, ystep = self.pixToSteps([xpix, ypix])

        xNew, yNew = self.moveToSteps(xstep, ystep, preload=preload, onlyIfNecessary=onlyIfNecessary)
        return self.stepsToPix([xNew, yNew])

    def getTargetPosition(self, wave, row):
        """Get the final x,y pixel position for a wave+row. Applies a .nudge if one exists. """

        led = self.lamps[self.lamps.wave == wave]
        col = int(led.position)
        nudgeKey = str((int(wave), int(row)))
        pos = (col, row)
        try:
            nudge = self.nudges[nudgeKey]
            self.logger.info(f'nudge for {nudgeKey}: {pos} to {nudge}')
            pos = nudge
        except KeyError:
            pass

        return pos

    def home(self, doX=True, doY=True):
        """Home one or more axes. Both by default. The controller leaves it at "middle" of range. """

        if doX:
            self.dev.cmd("home x", maxTime=100)
        if doY:
            self.dev.cmd("home y", maxTime=100)
        return self.getSteps()

def takeSuperDark(meade, nexp=3, nread=3, force=False):
    offtimeRequired = 3600
    offTime = meade.ledOffTime()
    cam = meade.cam
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
    if isinstance(comment, str):
        comment = comment.replace(' ', '_')
    pfsutils.oneCmd(f'hx_{cam}', f'ramp nread={nread} exptype={exptype} objname=\"{comment}\"',
                    quiet=quiet)
    visit = pathUtils.pathToVisit(pathUtils.lastRamp(cam=cam))

    return visit

lastFocus = None
def moveFocus(cam, piston):
    """Move the FPA focus. Honors any defined tilts.

    If we are not known to be moving positive, go 10um under the target and come back.

    Parameters
    ----------
    cam : `str`
        Camera name, like "n1"
    piston : `float`
        Absolute focus position, in microns.
    """

    global lastFocus

    if lastFocus is None or piston < lastFocus:
        pfsutils.oneCmd(f'xcu_{cam}', f'motors moveFocus microns={piston-10} abs')

    pfsutils.oneCmd(f'xcu_{cam}', f'motors moveFocus microns={piston} abs')
    lastFocus = piston

def motorScan(meade, xpos, ypos, led=None, call=None, nread=3, posInPixels=True):
    """Move to the given positions and acquire spots.

    This can be used to acquire dithers or a larger grid of spots.

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
        meade.lampsOff()

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

def createDither(frames, hxCalib, rad=15, doNorm=False, meade=None, r1=-1):
    """Create a dithered spot from 9 individual raw spots.

    Parameters
    ----------
    frames : `pd.DataFrame`
        The raw measures (esp xstep, ystep)
    hxCalib : `hxramp.HxCalib`
        Holds the bad pixel mask
    rad : int, optional
        What to use for background, by default 15
    doNorm : bool, optional
        Normalize flux to the 1st spot, by default False
    meade : `GimbalIlluminator`
        If the spot center is not known, what to ask for a good guess.
    r1 : int, optional
        The H4 read to use, by default -1

    Returns
    -------
    dither : image
        The composed dither image
    inIms : list of images
        The input image postage stamps
    outIms : list of images
        The dither component images.
    """
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

    if 'xpix' not in frames or np.isnan(frames.xpix.values[ctrIdx]):
        pixFromStep = True
        ctrPix = meade.stepsToPix((frames[['xstep','ystep']].values[ctrIdx]))
        ctr = np.round(ctrPix).astype('i4')
    else:
        pixFromStep = False
        ctr = np.round(frames[['xpix','ypix']].values[ctrIdx]).astype('i4')
    print(f'ctr={ctr}')
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
        print(f'{f1.visit:0.0f}: wave: {f1.wavelength} row: {f1.row} focus: {f1.focus} '
              f'pix: {xoff} {yoff} {pixFromStep} step: {f1.xstep:0.0f},{f1.ystep:0.0f} '
              f'bkgnd: {bkgnd:0.3f} '
              #f'ctr: {f1.xpix:0.2f},{f1.ypix:0.2f} bkgnd: {bkgnd:0.3f} '
              f'scale: {normSum:0.1f}/{imSum:0.1f}={normSum/imSum:0.3f}')
        outIm[yoff::scale, xoff::scale] = im
        inIms.append(im)

        out1 = outIm*0
        out1[yoff::scale, xoff::scale] = im
        outIms.append(out1)

    return outIm, inIms, outIms

def ditherPath(butler, row, pfsDay=None):
    if pfsDay is None:
        idDict = dict(visit=int(row.visit),
                      wavelength=int(row.wavelength),
                      focus=int(row.focus),
                      row=int(row.row))
        path = butler.getPath('dither', idDict=idDict)
        return path
    else:
        idDict = dict(visit=int(row.visit),
                      wavelength=int(row.wavelength),
                      focus=int(row.focus),
                      row=int(row.row))
        path = butler.search('dither', idDict=idDict,
                             pfsDay=pfsDay)
        return path[0]

def ditherPaths(butler, rows, pfsDay='*'):
    paths = []
    for i in range(0, len(rows), 9):
        row = rows.iloc[i]
        path = ditherPath(butler, row, pfsDay=pfsDay)[0]
        paths.append(path)
    return paths

def writeDither(row, butler, dithIm):
    """Save a dither and measurements to disk

    Parameters
    ----------
    row : `pd.DataFrame`
        The measurements for the dither
    butler : `nirander.Butler`
        Knows where to save files.
    dithIm : image
        The dither itself.
    """
    idDict = dict(visit=int(row.visit),
                  wavelength=int(row.wavelength),
                  focus=int(row.focus),
                  row=int(row.row))
    path = butler.get('dither', idDict=idDict)
    hdr = [dict(name='VISIT', value=int(row.visit), comment="visit of 0,0 image"),
            dict(name='WAVE', value=float(row.wavelength)),
            dict(name='FOCUS', value=float(row.focus)),
            dict(name='ROW', value=int(row.row), comment="rounded row number, for easy grouping")]
    try:
        hdr.extend([dict(name='XPIX0', value=float(row.xpix0), comment="calculated xc of 0,0 image"),
                    dict(name='YPIX0', value=float(row.ypix0), comment="calculated yc of 0,0 image")])
    except AttributeError:
        pass
    try:
        hdr.extend([dict(name='XPIX', value=float(row.xpix), comment="measured xc of 0,0 image"),
                    dict(name='YPIX', value=float(row.ypix), comment="measured yc of 0,0 image")])
    except AttributeError:
        pass
    try:
        hdr.extend([dict(name='SIZE', value=float(row.size), comment="measured RMS of 0,0 image"),
                    dict(name='X2', value=float(row.x2), comment="measured 2nd moment"),
                    dict(name='Y2', value=float(row.y2), comment="measured 2nd moment"),
                    dict(name='FLUX', value=float(row.flux), comment="measured total flux of 0,0 image"),
                    dict(name='PEAK', value=float(row.peak), comment="measured peak of 0,0 image")])
    except AttributeError:
        pass
    try:
        hdr.extend([dict(name='EE1', value=float(row.ee1), comment="EE of central pixel"),
                    dict(name='EE3', value=float(row.ee3), comment="EE of central 3 pixel box"),
                    dict(name='EE5', value=float(row.ee5), comment="EE of central 5 pixel box")])
    except AttributeError:
        pass

    # Damnit: FITS header cards do not support NaNs. Hack here.
    hdr0 = hdr
    hdr = []
    for h in hdr0:
        if not np.isnan(h['value']):
            hdr.append(h)

    path.parent.mkdir(parents=True, exist_ok=True)
    fitsio.write(path, dithIm, header=hdr, clobber=True)
    logger.info(f'wrote dither {path}')

    return idDict

def allDithers(frames, hxCalib, rad=15, butler=None, doNorm=False, meade=None, r1=-1):
    dithers = []
    ids = []
    for i in range(len(frames)//9):
        dithFrames = frames.iloc[i*9:(i+1)*9]
        print(len(dithFrames))
        try:
            dith1, _, _ = createDither(dithFrames, hxCalib, rad=rad, doNorm=doNorm, meade=meade,
                                       r1=r1)
        except Exception as e:
            print(f'failed to create dither from {dithFrames.visit.values[0]}: {e}')
            continue

        dithers.append(dith1)

        if butler is not None:
            row = dithFrames.iloc[0]
            idDict = writeDither(row, butler, dith1)
            ids.append(idDict)

    return pd.DataFrame(ids)

def ditherAt(meade, led, row, nramps=3, npos=3, nread=3, xsteps=5, ysteps=2):
    """Acquire dithered imaged at a given position. """

    if npos%2 != 1:
        raise ValueError("not willing to deal with non-odd dithering")
    rad = npos//2
    xc, yc = meade.pixToSteps([meade.lamps.position[led], row])
    x0, y0 = xc-(rad*xsteps), yc-(rad*ysteps)

    xx = x0 + np.arange(npos)*xsteps
    yy = y0 + np.arange(npos)*ysteps

    visits = []
    for r_i in range(nramps):
        gridVisits = motorScan(meade, xx, yy, led=led, nread=nread,
                               posInPixels=False)
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

def _setRowWindow(meade, row, windowWidth):
    """Configure H4 row skipping to straddle a row

    Parameters
    ----------
    meade : `GimbalIlluminator`
        Knows about the camera
    row : `int`
        The center row we want to read
    windowWidth : `int`
        The radius of the window to read
    """
    if row <= windowWidth-4 or row >= (4092-windowWidth):
        raise ValueError(f'row window too close to edge: {row}')

    skipToWindow = int(row) - windowWidth - 4
    skipToTopRef = 4092 - (int(row) + windowWidth)
    pfsutils.oneCmd(f'hx_{meade.cam}',
                    f'setRowSkipping skipSequence=4,{skipToWindow},{2*windowWidth},'
                    f'{skipToTopRef},{2*windowWidth + 8}')

def _loopByWaves(meade, butler, waves, rows, focus,
                 doDither=False, nread=3, doWindow=True, windowWidth=50):
    spotList = []
    for w_i, w in enumerate(waves):
        meade.led(w)
        _, dutyCycle, _ = meade.ledState()
        for r_i, row in enumerate(rows):
            pos = meade.getTargetPosition(w, row)
            if doWindow:
                _setRowWindow(meade, pos[1], windowWidth)
            if not doDither:
                meade.moveToPix(*pos, preload=True)
            for f_i, f in enumerate(focus):
                print(f"led {w} on row {int(row)} with focus {f}")
                moveFocus(meade.cam, f)
                try:
                    if doDither:
                        meas = ditherAtPix(meade, pos=pos, nread=nread)
                    else:
                        meas = takeBareSpot(meade, nread=nread,
                                            comment=f'spotSet_{w}_{round(row)}_{round(f)}')
                except Exception as e:
                    raise

                meas['row'] = int(row)
                meas['focus'] = f
                meas['wavelength'] = w
                meas['dutyCycle'] = dutyCycle
                spotList.append(meas)

                rowFrame = pd.concat(spotList, ignore_index=True)
                if butler is not None:
                    outFileName = writeRawMeasures(butler, rowFrame)
                    print(f"wrote {len(rowFrame)} lines to {outFileName} "
                            f"at led {w} on row {row} with focus {f}")
    return spotList

def _loopByFocus(meade, butler, focus, rows, waves,
                 doDither=False, nread=3, doWindow=True, windowWidth=50):
    spotList = []
    for f_i, f in enumerate(focus):
        moveFocus(meade.cam, f)
        for r_i, row in enumerate(rows):
            # Any wavelength should do fine.
            pos = meade.getTargetPosition(waves[0], row)
            if doWindow:
                _setRowWindow(meade, pos[1], windowWidth)
            for w_i, w in enumerate(waves):
                pos = meade.getTargetPosition(w, row)
                meade.led(w)
                _, dutyCycle, _ = meade.ledState()
                print(f"led {w} on row {int(row)} with focus {f}")
                try:
                    if doDither:
                        meas = ditherAtPix(meade, pos=pos, nread=nread)
                    else:
                        meade.moveToPix(*pos, preload=True, onlyIfNecessary=True)
                        meas = takeBareSpot(meade, nread=nread,
                                            comment=f'spotSet_{w}_{round(row)}_{round(f)}')
                except Exception as e:
                    raise

                meas['row'] = int(row)
                meas['focus'] = f
                meas['wavelength'] = w
                meas['dutyCycle'] = dutyCycle
                spotList.append(meas)

                rowFrame = pd.concat(spotList, ignore_index=True)
                if butler is not None:
                    outFileName = writeRawMeasures(butler, rowFrame)
                    print(f"wrote {len(rowFrame)} lines to {outFileName} "
                            f"at led {w} on row {row} with focus {f}")

    return spotList

def _loopOverPos(meade, butler, focus, posList,
                 doDither=False, nread=3, doWindow=True, windowWidth=50):
    spotList = []

    for f_i, f in enumerate(focus):
        moveFocus(meade.cam, f)

        for w, row in posList:
            pos = meade.getTargetPosition(w, row)
            if doWindow:
                _setRowWindow(meade, pos[1], windowWidth)
            meade.led(w)
            _, dutyCycle, _ = meade.ledState()
            print(f"led {w} on row {int(row)} with focus {f}")
            try:
                if doDither:
                    meas = ditherAtPix(meade, pos=pos, nread=nread)
                else:
                    meade.moveToPix(*pos, preload=True, onlyIfNecessary=True)
                    meas = takeBareSpot(meade, nread=nread,
                                        comment=f'spotSet_{w}_{round(row)}_{round(f)}')
            except Exception as e:
                raise

            meas['row'] = int(row)
            meas['focus'] = f
            meas['wavelength'] = w
            meas['dutyCycle'] = dutyCycle
            spotList.append(meas)

            rowFrame = pd.concat(spotList, ignore_index=True)
            if butler is not None:
                outFileName = writeRawMeasures(butler, rowFrame)
                print(f"wrote {len(rowFrame)} lines to {outFileName} "
                        f"at led {w} on row {row} with focus {f}")

    return spotList

def spotSet(meade, butler=None, waves=None, rows=None, posList=None, focus=None,
            doDither=False, nread=3, doWindow=True, windowWidth=50,
            byFocus=True):
    """Primary acquisition routine: takes a (wave, row, focus) grid of spots or dithers

    Parameters
    ----------
    meade : `GimbalIlluminator`
        The object which controls lamps and moves the gimbalator.
    butler : `butler.Butler`, optional
        Path/file wrapper, by default None
    waves : float or list of floats, optional
        wavelength to take spots at. By default uses all the defined lamps.
    rows : float or list of floats
        rows to take stops at. Not really optional.
    posList : list of (wave, row) pairs
        In PLACE of waves and rows.
    focus : float or list of floats
        FPA focus position to take spots at. Not really optional
    doDither : bool, optional
        Whether to take a 3x3 point dither, by default False
    nread : int, optional
        Numbers of H4 reads to take per spot, by default 3
    doWindow : bool, optional
        Whether to window using the H4 row skipping option, by default False
    windowWidth : int, optional
        If doWindow=True, the "radius" of the window, by default 50.
    byFocus : bool, optional
        If True, loop first by focus positions, to minimize FPA motor moves

    Returns
    -------
    spotFrame : `pd.DataFrame`
        The essentials for the acquisition: visit, motors steps, focus, nominal row, wavelength.

    The spotFrame is saved to a butler-curated location.

    """

    if posList is not None:
        if (waves is not None or rows is not None):
            raise RuntimeError("if posList specified, waves and/or rows cannot be.")
    else:
        if waves is None:
            waves = meade.lamps.wave
        if np.isscalar(waves):
            waves = [waves]

        if rows is None:
            raise ValueError("rows must be specified, either a scalar or a list of positions.")
        if np.isscalar(rows):
            rows = [rows]
        rows = np.array(rows, dtype='f4')

    if focus is None:
        raise ValueError("focus must be specified, either a scalar or a list of values.")
    if np.isscalar(focus):
        focus = [focus]
    focus = np.array(focus, dtype='f4')

    if len(focus) == 1:
        moveFocus(meade.cam, focus[0])

    try:
        if posList is not None:
            spotList = _loopOverPos(meade, butler, focus, posList,
                                    doDither=doDither, nread=nread,
                                    doWindow=doWindow, windowWidth=windowWidth)
        elif byFocus:
            spotList = _loopByFocus(meade, butler, focus, rows, waves,
                                    doDither=doDither, nread=nread,
                                    doWindow=doWindow, windowWidth=windowWidth)
        else:
            spotList = _loopByWaves(meade, butler, waves, rows, focus,
                                    doDither=doDither, nread=nread,
                                    doWindow=doWindow, windowWidth=windowWidth)
    except Exception as e:
        print(f'oops: {e}')
        # breakpoint()
        raise
    finally:
        meade.lampsOff()
        if doWindow:
            pfsutils.oneCmd(f'hx_{meade.cam}', 'clearRowSkipping')

    return pd.concat(spotList, ignore_index=True)

def _writeMeasures(butler, df, measureType):
    outFileName = butler.getPath(measureType,
                                 idDict=dict(visit=int(df.visit.min())))
    outFileName.parent.mkdir(mode=0o2775, parents=True, exist_ok=True)
    with open(outFileName, mode='w') as outf:
        outf.write(df.to_string())

    return outFileName

def writeRawMeasures(butler, df):
    return _writeMeasures(butler, df, 'rawMeasures')

def writeSpotMeasures(butler, df):
    return _writeMeasures(butler, df, 'measures')

def writeDitherMeasures(butler, df):
    return _writeMeasures(butler, df, 'ditherMeasures')

def trimRect(im, c, r=100):
    cx, cy = c
    im2 = im[cy-r:cy+r, cx-r:cx+r]

    return im2.copy()

def rectMask(mask, center, radius=100):
    """zero out pixels rectangularily further than radius from ctr. """

    newMask = mask.copy()
    ctrX = round(center[0])
    ctrY = round(center[1])
    preCount = mask.sum()
    newMask[:max(0, (ctrY-radius)), :] = 1
    newMask[min(4095, ctrY+radius):, :] = 1
    newMask[:, :max(0, (ctrX-radius))] = 1
    newMask[:, min(4095, (ctrX+radius)):] = 1
    # logger.debug(f'{ctrX}, {ctrY}, {radius} {preCount} {newMask.sum()}')

    return newMask

def getPeaks(im, thresh=250.0, mask=None, center=None, radius=10,
             searchRadius=5,
             convolveSigma=None, kernel=True):

    if center is not None and mask is not None:
        mask = rectMask(mask, center, radius)
    bkgnd = sep.Background(im.astype('f4'), mask=mask)
    bkg = bkgnd.back()
    corrImg = im - bkg

    if convolveSigma is not None:
        corrImg = scipy.ndimage.gaussian_filter(corrImg, sigma=convolveSigma)

    # Doing this the expensive way: extract on full image, then trim
    try:
        argDict = dict(deblend_cont=1.0,
                       thresh=thresh,
                       mask=mask)
        if kernel is not True:
            argDict['filter_kernel'] = kernel

        spots = sep.extract(corrImg, **argDict)
    except Exception as e:
        logger.warning(f'getPeaks failed: {e}')
        return corrImg, None

    spotsFrame = pd.DataFrame(spots)
    spotsFrame['ellipticity'] = spotsFrame.x2 / spotsFrame.y2
    spotsFrame.loc[spotsFrame.ellipticity < 1,
                   'ellipticity'] = 1/spotsFrame.loc[spotsFrame.ellipticity < 1,
                                                     'ellipticity']
    if center is not None:
        center = np.atleast_2d(center)
        keep_w = cdist(spotsFrame[["x","y"]], center) <= searchRadius
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

def getPolyMin(sweep, xname, yname):
    sweep = sweep.dropna()
    x = sweep.sort_values(by=xname)[xname]
    y = sweep.sort_values(by=xname)[yname]

    if len(x) == 1:
        minx = x.values[0]
        poly = np.polynomial.Polynomial([y.values[0]/x.values[0]])
    else:
        poly = np.polynomial.Polynomial.fit(x,y,min(2, len(x)))
        c,b,a = poly.convert().coef
        minx = -b/(2*a)
    return minx, poly

def getBestFocus(sweep):
    return getPolyMin(sweep, 'focus', 'size')

def getFocusGrid(center, spacing=2, r=5):
    focusReq = center + (np.arange(2*r-1) - (r-1))*spacing
    return focusReq

def focusSweep(meade, butler, wave, row,
               center, spacing, r,
               nread=3, measureCall=None, doWindow=True):
    focusReq = getFocusGrid(center, spacing=spacing, r=r)
    print(focusReq)

    if focusReq[0] < 15:
        raise RuntimeError(f"focusReq[0] too low, not starting below: focusReq")

    moveFocus(meade.cam, focusReq[0]-10)

    scanFrame = spotSet(meade, butler, waves=[wave], rows=[row],
                        focus=focusReq, doWindow=doWindow,
                        nread=nread)
    if measureCall is not None:
        try:
            focusSet = measureCall(scanFrame)
            bestFocus, focusPoly = getBestFocus(focusSet)
            print(f"best focus: {bestFocus}")
            if (bestFocus is not None and
                bestFocus >= focusReq[0] and
                bestFocus <= focusReq[-1]):

                bestFrame = spotSet(meade, butler, waves=[wave], rows=[row],
                                    focus=bestFocus, doWindow=doWindow)
                bestSize = focusPoly(bestFocus)
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
    xpix0, ypix0 = meade.stepsToPix((xstep, ystep))
    scanFrame['wavelength'] = wavelength
    scanFrame['xstep'] = xstep
    scanFrame['ystep'] = ystep
    scanFrame['dutyCycle'] = dutyCycle
    scanFrame['xpix0'] = xpix0
    scanFrame['ypix0'] = ypix0

    return scanFrame

def measureSet(scans, meade=None, hxCalib=None, thresh=10, center=None,
               radius=10, searchRadius=5, skipDone=True, ims=None, trimBad=True, doClear=False,
               convolveSigma=None, kernel=True, remask=False,
               rawSpots=False, r0=0, r1=-1):
    """Measure the best spots in a DataFrame of images

    Parameters
    ----------
    scans : `pd.DataFrame`
        DataFrame with at least visit,xstep,ystep.
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
        hxCalib = hxcalib.HxCalib()

    for f in 'x2', 'y2', 'xpix', 'ypix', 'flux', 'peak', 'size':
        if f not in scans or doClear:
            scans[f] = np.nan

    if skipDone:
        notDone = scans[scans.xpix.isna()].index
    else:
        notDone = scans.index

    for i_i, scan_i in enumerate(notDone):
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
            corrImg = ims[i_i]
            if center is None:
                center_i = None
        else:
            if hxCalib is not None:
                corrImg = hxCalib.isr(scans.loc[scan_i, 'visit'], r0=r0, r1=r1)
#                if remask:
#                    path = hxramp.rampPath(visit=scans.loc[scan_i, 'visit'])
#                    data0 = hxRamp.HxRamp(path).dataN(0)

            else:
                ramp = hxramp.HxRamp(visit=scans.loc[scan_i, 'visit'])
                corrImg = ramp.cdsN(r0=r0, r1=r1)

        try:
            corrImg, spots = getPeaks(corrImg,
                                      center=center_i, radius=radius,
                                      searchRadius=searchRadius,
                                      thresh=thresh,
                                      mask=hxCalib.badMask,
                                      convolveSigma=convolveSigma,
                                      kernel=kernel)
        except Exception as e:
            logger.warning(f'getPeaks failed: {e}')
            corrImg = None
            spots = []

        if trimBad and len(spots) > 0:
            ok = (spots.size < 150) & (spots.ellipticity < 5.0)
            origSpots = spots.copy()
            spots = spots[ok]
        if spots is None or len(spots) == 0:
            print(f"nope: i={scan_i}, scan={scans.loc[scan_i]}")
        else:
            # remeasure, using the windowed routines:
            # sep.winpos()
            bestSpot = spots.loc[spots.flux.idxmax()]
            print(f"    : i={scan_i}, visit={scans.loc[scan_i, 'visit']}, nspots={len(spots)} ({bestSpot.x:0.2f}, {bestSpot.y:0.2f})")
            scans.loc[scan_i, 'xpix'] = bestSpot.x
            scans.loc[scan_i, 'ypix'] = bestSpot.y
            scans.loc[scan_i, 'x2'] = bestSpot.x2
            scans.loc[scan_i, 'y2'] = bestSpot.y2
            scans.loc[scan_i, 'xy'] = bestSpot.xy
            scans.loc[scan_i, 'size'] = 2*np.sqrt(bestSpot.x2 + bestSpot.y2)
            scans.loc[scan_i, 'a'] = bestSpot.a
            scans.loc[scan_i, 'b'] = bestSpot.b
            scans.loc[scan_i, 'size_ab'] = 2*np.sqrt(bestSpot.a + bestSpot.b)
            scans.loc[scan_i, 'flux'] = bestSpot.flux
            scans.loc[scan_i, 'peak'] = bestSpot.peak
            scans.loc[scan_i, 'xmin'] = bestSpot.xmin
            scans.loc[scan_i, 'xmax'] = bestSpot.xmax
            scans.loc[scan_i, 'ymin'] = bestSpot.ymin
            scans.loc[scan_i, 'ymax'] = bestSpot.ymax

    return scans

def takeBareSpot(meade, nread=3, comment="no_comment"):
    """Lowest-level exposure which returns a dataframe with (visit, xstep, ystep, led) """

    visit = takeRamp(cam=meade.cam, nread=nread, exptype='object', comment=comment)
    df = basicDataFrame(meade, visits=[visit])

    return df

def takeSpot(meade, pos=None, focus=None, light=None, nread=3, comment="no_comment"):
    """Lowest-level exposure which optionally sets focus/led/gimbal and returns a dataframe. """

    if pos is not None:
        meade.moveToPix(*pos)
    if focus is not None:
        moveFocus(meade.cam, focus)
    if light is not None:
        if np.isscalar(light):
            meade.led(light)
        else:
            meade.led(*light)
    df = takeBareSpot(meade, nread=nread, comment=comment)
    if focus is not None:
        df['focus'] = focus

    if light is not None:
        meade.lampsOff()
    return df

def oneSpotFromDither(df):
    """Return a single spot (the LL one) from each dither in a frame"""
    rows = []
    for gname, grp in df.groupby(['focus', 'row', 'wavelength']):
        rows.append(grp[(grp.xstep == grp.xstep.min()) & (grp.ystep == grp.ystep.min())])
    return pd.concat(rows)
