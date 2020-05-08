from importlib import reload
import logging
import socket
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.transform

import sep
from matplotlib.patches import Ellipse
from scipy.spatial.distance import cdist

from pfs.utils import butler as pfsButler
from pfs.utils import spectroIds as pfsSpectroIds

from . import butlerMaps
from . import darkCube
from . import hxstack as hx
from . import pfsutils

reload(pfsButler)
reload(pfsSpectroIds)
reload(butlerMaps)
reload(darkCube)
reload(hx)
reload(pfsutils)

specIds = pfsSpectroIds.SpectroIds(partName='n1')
nirButler = pfsButler.Butler(specIds=specIds, 
                             configRoot=butlerMaps.configKeys['nirLabConfigRoot'])
nirButler.addMaps(butlerMaps.configMap, butlerMaps.dataMap)
nirButler.addKeys(butlerMaps.configKeys)

scaleDtype = np.dtype([('index', '<i4'), 
                       ('visit', '<i4'), 
                       ('xstep', '<i4'), 
                       ('ystep', '<i4'), 
                       ('xpix', '<f4'), 
                       ('ypix', '<f4')])

projectionCoeffs = np.array([[-7.29713459e-02,  5.03186582e-03,  4.18365204e+03],
                             [ 2.08787075e-04,  1.91548157e-01, -6.78352586e+02],
                             [ 2.37363712e-08,  2.42779233e-06,  9.64776455e-01]])
stepToPix = skimage.transform.ProjectiveTransform(projectionCoeffs)
pixToStep = stepToPix.inverse

class NirIlluminator(object):
    def __init__(self, forceLedOff=True, logLevel=logging.INFO):
        self.logger = logging.getLogger('meade')
        self.logger.setLevel(logLevel)
        self._led = None
        self._ledPower = None
        self._ledChangeTime = None

        # Ordered by increasing X _steps_, decreasing X _pixels_ (why did I say yes?!?)
        self.leds = pd.DataFrame(dict(wave=[1300, 1200, 1085, 1070, 1050, 970, 930],
                                      dutyCycle=[100.0, 33, 30, 33, 19, 83, 40],
                                      focusOffset=[4.0, 0, 0, 0, 0, 0, -10.0],
                                      position=[3984, 3664, 2700, 2457, 2274, 846, 100]))
        self.leds = self.leds.set_index('wave', drop=False)

        self.preloadDistance = 200
        self.motorSlips = (0, 0)
        
        if forceLedOff:
            self.ledsOff()
        
    def __str__(self):
        return f"Meade(led={self._led}@{self._ledPower}, steps={self.getSteps()}, pix={self.getPix()})"
    def __repr__(self):
        return self.__str__()
    
    def _cmd(self, cmdStr, debug=False, maxTime=5.0):
        """ Send a single motor command.
        
        Args
        ----
        cmdStr : str
            Either "move x y" or a raw AllMotion command. "move" commands
            are checked against internal limit, so should always be used to 
            move.
        """
        ip = '192.168.1.12'
        port = 9999
        
        if debug:
            logFunc = self.logger.warning
        else:
            logFunc = self.logger.debug
            
        cmdStr = cmdStr + '\n'
        replyBuffer = ""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((ip, port))
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
            
        parts = replyBuffer.split('\n')
        return parts[0]

    @property
    def dutyCycle(self):
        return self._ledPower
        
    def ledState(self):
        if self._ledChangeTime is None:
            dt = None
        else:
            dt = time.time() - self._ledChangeTime
        return (self._led, self._ledPower, dt)

    def ledPosition(self, y, led=None):
        # Ignores Y, which is wrong -- CPL

        if led is None:
            led = self._led
 
        return self.leds.position[led]
 
    def ledFocusOffset(self, y, led=None):
        # Ignores Y, which is wrong -- CPL

        if led is None:
            led = self._led
 
        return self.leds.loc[led]['focusOffset']

    def ledOffTime(self):
        _led, _ledPower, dt = self.ledState()
        
        if _ledPower != 0:
            return 0
        else:
            return dt
        
    def ledsOff(self):
        for w in self.leds.wave:
            self._cmd(f'led {w} 0')
        self._led = 0
        self._ledPower = 0
        self._ledChangeTime = time.time()
        
    def led(self, wavelength, dutyCycle=None):
        wavelength = int(wavelength)
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
            self._cmd(f'led {self._led} 0', debug=True)
            self._led = 0
        self._led = wavelength
        self._ledPower = dutyCycle
        self._ledChangeTime = time.time()
        
        self._cmd(f'led {self._led} {dutyCycle}', debug=True)
        
    def stepsToPix(self, steps):
        steps = np.array(steps)
        upDim = steps.ndim < 2
        if upDim:
            steps = np.atleast_2d(steps)
        pix = stepToPix(steps)
        return pix[0] if upDim else pix

    def pixToSteps(self, pix):
        pix = np.array(pix)
        upDim = pix.ndim < 2
        if upDim:
            pix = np.atleast_2d(pix)
        steps = np.round(pixToStep(pix)).astype('i4')
        return steps[0] if upDim else steps

    def getSteps(self):
        xPos = int(self._cmd('/1?0'))
        yPos = int(self._cmd('/2?0'))
        
        return xPos, yPos

    def getPix(self):
        xSteps, ySteps = self.getSteps()
        return self.stepsToPix([xSteps, ySteps])
    
    def moveSteps(self, dx, dy):
        xPos, yPos = self.getSteps()
        self.moveTo(xPos+dx, yPos+dy)
        
    def moveTo(self, x, y, preload=True):
        x = int(x)
        y = int(y)
        xPos, yPos = self.getSteps()
        if preload:
            cmdStr = f"move {x-self.preloadDistance} {y-self.preloadDistance}"
            self._cmd(cmdStr, debug=True, maxTime=45)
            
        dist = max(abs(x-xPos), abs(y-yPos))
        cmdStr = f"move {x} {y}"
        self._cmd(cmdStr, debug=True, maxTime=dist/1000)
        
        xNew, yNew = self.getSteps()
        
        if x != xNew or y != yNew:
            raise RuntimeError(f'did not move right: target={x},{y}, at={xNew},{yNew}')
            
        return xNew, yNew
            
    def moveToPix(self, xpix, ypix, preload=True):
        xstep, ystep = self.pixToSteps([xpix, ypix])
        
        xNew, yNew = self.moveTo(xstep, ystep, preload=preload)
        return self.stepsToPix([xNew, yNew])
        
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

def takeRamp(cam, nread):
    pfsutils.oneCmd(f'hx_{cam}', f'ramp nread={nread}')
    visit = hx.pathToVisit(hx.lastRamp(cam=cam))
    
    return visit

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
            xpos = meade.leds.position[wavelength]
    res = []
    
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
            meade.moveTo(xStep, yStep, preload=preload)
            lastXStep = xStep
            lastYStep = yStep

            if call is not None:
                ret = call(meade)
                callRet.append(ret)
            else:
                visit = takeRamp('n1', nread=nread)
                res.append([visit, xStep, yStep])

    if led is not None:
        meade.ledsOff()

    if call is None:
        return [pd.DataFrame(res, columns=('visit', 'xstep', 'ystep'))]
    else:
        return callRet

def checkMoveRepeats(meade, start=(2000,2000), nrep=10, npos=5, pixStep=2):
    xrepeats = []
    yrepeats = []
    far = []
    
    rng = np.random.default_rng(2394)

    x0, y0 = start
    xx = np.arange(x0, x0 + npos*pixStep+1, pixStep)
    yy = np.arange(y0, y0 + npos*pixStep+1, pixStep)
    
    for i in range(nrep):
        farx, fary = rng.uniform(0,4096), rng.uniform(0,4096)
    
        tscan = motorScan(meade, farx, fary, led=(1085,30))
        far.append(tscan)
        tscan = motorScan(meade, xx, y0, led=(1085,30))
        xrepeats.append(tscan)
    
        tscan = motorScan(meade, farx, fary, led=(1085,30))
        far.append(tscan)
        tscan = motorScan(meade, x0, yy, led=(1085,30))
        yrepeats.append(tscan)
        
    measureSet(meade, far)
    measureSet(meade, xrepeats)
    measureSet(meade, yrepeats)

    return xrepeats, yrepeats, far

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

def ditherAt(meade, led, row, nramps=3, npos=3, nread=3, xsteps=5, ysteps=2):
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

def ditherSet(meade, butler=None, waves=None, rows=[88,2040,3995], focus=122.0,
              nramps=3):
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
            meade.led(w)
            led, dutyCycle, _ = meade.ledState()
            for r_i, row in enumerate(rows):
                for f_i, f in enumerate(focus):
                    print(f"led {w} on row {row} with focus {f}")
                    pfsutils.oneCmd('xcu_n1', f'motors move piston={f} abs microns')
                    try:
                        ret = ditherAt(meade, w, row, nramps=nramps)
                    except Exception as e:
                        breakpoint()

                    ret['focus'] = f
                    ret['wavelength'] = w
                    ret['dutyCycle'] = dutyCycle
                    ditherList.append(ret)
                    
                    print("ditherList: ", ditherList)
                    rowFrame =  pd.concat(ditherList, ignore_index=True)
                    if butler is not None:
                        outFileName = butler.get('measures', idDict=dict(visit=rowFrame.visit.min()))
                        outFileName.parent.mkdir(mode=0o2775, parents=True, exist_ok=True)
                        with open(outFileName, mode='w') as outf:
                            outf.write(rowFrame.to_string())
                            print(f"wrote {len(rowFrame)} lines to {outFileName} at led {w} on row {row} with focus {f}")
    except Exception as e:
        print(f'oops: {e}')
        breakpoint()
        raise
    finally:
        meade.ledsOff()
        return ditherList

def trimRect(im, c, r=100):
    cx, cy = c
    im2 = im[cy-r:cy+r, cx-r:cx+r]
    
    return im2.copy()
    
def getPeaks(im, thresh=250.0, mask=None, center=None, radius=100):
    bkgnd = sep.Background(im, mask=mask)
    bkg = np.array(bkgnd)
    corrImg = im - bkg
    
    # Doing this the expensive way: extract on full image, then trim
    spots = sep.extract(corrImg, deblend_cont=1.0, 
                        thresh=thresh, mask=mask)
    spotsFrame = pd.DataFrame(spots)
    if center is not None:
        keep_w = cdist(spotsFrame[["x","y"]], center) <= radius
        spotsFrame = spotsFrame.loc[keep_w]
        if len(spotsFrame) != 1:
            print(f'got {len(spotsFrame)} spots near {center}')
            spotsFrame = None

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
        visit = takeRamp(cam=cam, nread=nread)
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
                visit = takeRamp(cam=cam, nread=nread)
                visits.append(visit)

                bestSize = focusPoly(bestFocus)
                bestFrame = pd.DataFrame(dict(visit=[visit], focus=[bestFocus]))
                measureCall(bestFrame)
                print(f"expected {bestSize:0.2f}, got {bestFrame['size'].values[0]:0.2f} ")

                scanFrame = pd.concat([scanFrame, bestFrame], ignore_index=True)
        except Exception as e:
            print(f"Failed to measure and go to best focus: {e}")

    return scanFrame
            

def scanForFocus(center, spacing=5, r=4, measureCall=None):
    return _scanForFocus(center, spacing=spacing, r=r, measureCall=measureCall)
def scanForCrudeFocus(center, spacing=25, r=3, measureCall=None):
    return _scanForFocus(center, spacing=spacing, r=r, measureCall=measureCall)

def measureSet(scans, hxCalib, thresh=250, center=None, radius=100, skipDone=False):
    """Measure the best spots in a DataFrame of images
    
    Parameters
    ----------
    scans : `pd.DataFrame`
        [description]
    hxCalib : `HxCalib`
        [description]
    thresh : int, optional
        [description], by default 250
    center : [type], optional
        [description], by default None
    radius : int, optional
        [description], by default 100
    skipDone : bool, optional
        [description], by default True
    
    Returns
    -------
    [type]
        [description]
    """
       
    for f in 'x2', 'y2', 'xpix', 'ypix', 'flux', 'peak', 'size':
        if f not in scans:
            scans[f] = np.nan

    if skipDone:
        notDone = scans[scans.xpix.isna()].index
    else:
        notDone = scans.index

    for scan_i in notDone:        
        if center is None:
            pixCenter = [scans.loc[scan_i, 'xstep'], scans.loc[scan_i, 'ystep']]
            center_i = stepToPix(pixCenter)
            # print(f"{scan_i} center={center} from {pixCenter}")
        else:
            center_i = center
            
        corrImg, spots = getPeaks(hxCalib.isr(scans.loc[scan_i, 'visit']),
                                  center=center_i, radius=radius,
                                  thresh=thresh, 
                                  mask=hxCalib.badMask)
        if spots is None:
            print(f"nope: i={scan_i}, scan={scans.loc[scan_i]}")
        if spots is not None:
            print(f"    : i={scan_i}, visit={scans.loc[scan_i, 'visit']}")
            bestSpot = spots.loc[spots.flux.idxmax()]
            scans.loc[scan_i, 'xpix'] = bestSpot.x
            scans.loc[scan_i, 'ypix'] = bestSpot.y
            scans.loc[scan_i, 'x2'] = bestSpot.x2
            scans.loc[scan_i, 'y2'] = bestSpot.y2
            scans.loc[scan_i, 'size'] =  (bestSpot.x2 + bestSpot.y2)/2
            scans.loc[scan_i, 'flux'] = bestSpot.flux
            scans.loc[scan_i, 'peak'] = bestSpot.peak
    
    return scans
