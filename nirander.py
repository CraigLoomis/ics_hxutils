from importlib import reload
import logging
import socket
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sep
from matplotlib.patches import Ellipse
from scipy.spatial.distance import cdist

from pfs.utils import butler as pfsButler
from pfs.utils import spectroIds as pfsSpectroIds

import butlerMaps
import hxstack as hx
import pfsutils

reload(pfsButler)
reload(pfsSpectroIds)
reload(butlerMaps)
reload(hx)
reload(pfsutils)

specIds = pfsSpectroIds.SpectroIds(partName='n1')
butler = pfsButler.Butler(specIds=specIds, configRoot=butlerMaps.nirLabConfigRoot)
butler.addMaps(butlerMaps.configMap, butlerMaps.dataMap)
butler.addKeys(dict(nirLabReduxRoot=butlerMaps.nirLabReduxRoot,
                    nirLabConfigRoot=butlerMaps.nirLabConfigRoot))

scaleDtype = np.dtype([('index', '<i4'), 
                       ('visit', '<i4'), 
                       ('xstep', '<i4'), 
                       ('ystep', '<i4'), 
                       ('xpix', '<f4'), 
                       ('ypix', '<f4')])

yscaleData = np.rec.array([(0, 5274, 30000,  5000, 2032.68227599,  306.15124233),
                           (1, 5275, 30000,  7000, 2032.78224718,  688.29252832),
                           (2, 5276, 30000,  9000, 2033.03649178, 1069.75591912),
                           (3, 5277, 30000, 11000, 2033.04260401, 1447.24120791),
                           (4, 5278, 30000, 13000, 2033.08429587, 1821.14349581),
                           (5, 5279, 30000, 15000, 2032.84115206, 2190.7910757 ),
                           (6, 5280, 30000, 17000, 2032.56526217, 2560.40970008),
                           (7, 5281, 30000, 19000, 2031.84808211, 2927.57128883),
                           (8, 5282, 30000, 21000, 2031.00032165, 3292.09321345),
                           (9, 5283, 30000, 23000, 2030.33943306, 3657.17707588)],
                          dtype=scaleDtype)
xscaleData = np.rec.array([(0, 5284,  6000, 14500,  290.81248957, 2097.45093701),
                           (1, 5285, 11000, 14500,  650.55960066, 2097.41935155),
                           (2, 5286, 16000, 14500, 1011.35055024, 2098.14972351),
                           (3, 5287, 21000, 14500, 1371.11712007, 2098.94319438),
                           (4, 5288, 26000, 14500, 1732.90726888, 2099.59015068),
                           (5, 5289, 31000, 14500, 2094.95596687, 2100.35489347),
                           (6, 5290, 36000, 14500, 2457.62415515, 2101.22703119),
                           (7, 5291, 41000, 14500, 2820.0307044 , 2102.12862705),
                           (8, 5292, 46000, 14500, 3183.40556648, 2102.88050178),
                           (9, 5293, 51000, 14500, 3548.148925  , 2103.49876305)],
                          dtype=scaleDtype)

# Forward and reverse maps. Assumed basically orthogonal nd quardratic
# Have seen ~10-pix shift in x. I think steps slipped on move towards 0, so 
#   am adding an offset.
#
xPixToStep = np.polynomial.Polynomial.fit(xscaleData.xpix, xscaleData.xstep, 2)
yPixToStep = np.polynomial.Polynomial.fit(yscaleData.ypix, yscaleData.ystep, 2)
xStepToPix = np.polynomial.Polynomial.fit(xscaleData.xstep, xscaleData.xpix, 2)
yStepToPix = np.polynomial.Polynomial.fit(yscaleData.ystep, yscaleData.ypix, 2)

class NirIlluminator(object):
    def __init__(self, forceLedOff=True, logLevel=logging.INFO):
        self.logger = logging.getLogger('meade')
        self.logger.setLevel(logLevel)
        self._led = None
        self._ledPower = None
        self._ledChangeTime = None
        
        self.wavelengths = {930, 970, 1050, 1070, 1085, 1200, 1300}
        
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
        ip = '192.168.1.198'
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
        
    def ledsOff(self):
        for w in self.wavelengths:
            self._cmd(f'led {w} 0')
        self._led = 0
        self._ledPower = 0
        self._ledChangeTime = time.time()
        
    def led(self, wavelength, dutyCycle):
        if wavelength not in self.wavelengths:
            raise ValueError(f"wavelength ({wavelength}) no in {self.wavelengths}")
        if dutyCycle < 0 or dutyCycle > 100:    
            raise ValueError(f"dutyCycle ({dutyCycle}) not in 0..100")
            
        if self._led is None:
            raise RuntimeError("current state of LEDs is unknown: need to call .ledsOff() before turning a LED on.")
        if self._led in self.wavelengths and self._led != wavelength:
            self._cmd(f'led {self._led} 0')
            self._led = 0
        self._led = wavelength
        self._ledPower = dutyCycle
        self._ledChangeTime = time.time()
        
        self._cmd(f'led {self._led} {dutyCycle}')
        
    def stepsToPix(self, xSteps, ySteps):
        return xStepToPix(xSteps), yStepToPix(ySteps)
    def pixToSteps(self, xPix, yPix):
        return int(round(xPixToStep(xPix))), int(round(yPixToStep(yPix)))

    def getSteps(self):
        xPos = int(self._cmd('/1?0'))
        yPos = int(self._cmd('/2?0'))
        
        return xPos, yPos

    def getPix(self):
        xSteps, ySteps = self.getSteps()
        return self.stepsToPix(xSteps, ySteps)
    
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
        xstep, ystep = self.pixToSteps(xpix, ypix)
        
        xNew, yNew = self.moveTo(xstep, ystep, preload=preload)
        return self.stepsToPix(xNew, yNew)
        
def takeSuperDark(meade, nexp=3, nread=3, force=False, cam='n1'):
    offtimeRequired = 3600
    offTime = meade.ledOffTime()
    if offTime < offtimeRequired:
        if not force:
            raise RuntimeError(f"need lamps to be off for {offtimeRequired}, not {offTime} seconds")

    paths = []
    visits = []
    for i in range(nexp):
        pfsutils.oneCmd(f'hx_{cam}', f'ramp nread={nread}',quiet=False)
        path = hx.lastRamp(cam=cam)
        paths.append(path)
        visits.append(pfsutils.pathToVisit(path))
        
    superDark = hx.medianCubes(visits)
        
    return superDark, visits

def takeRamp(cam, nread):
    pfsutils.oneCmd(f'hx_{cam}', f'ramp nread={nread}')
    visit = pfsutils.pathToVisit(hx.lastRamp(cam=cam))
    
    return visit

def motorScan(meade, xpos, ypos, led=None, call=None, nread=3, nramp=1, posInPixels=True):
    """Move to the given positions and acquire spots.
    
    Args
    ----
    xpos, ypos : number or array
      The positions to acquire at. Iterates over x first.
      If one position is a scalar, is expanded to match the other axis.
    led : (wavelength, dutyCycle)
      What lamp to turn on. If passed in, lamp is turned off at end of scan.
    nread : int
      How many reads to take per ramp
    nramp : int
      How many ramps to take at each position
    posInPixels : bool
      Whether `xpos` and `ypos` are in pixels or steps.
    
    ALWAYS sorts the positions and moves below the first position
    before acquiring data. 
    """
    
    if led is not None:
        wavelength, dutyCycle = led
        meade.led(wavelength=wavelength, dutyCycle=dutyCycle)

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
    for x_i, x in enumerate(xpos):
        for y_i, y in enumerate(ypos):
            if y_i == 0 and (x_i == 0 or len(ypos) > 1):
                preload = True
            else:
                preload = False
            if posInPixels:
                x1, y1 = meade.pixToSteps(x, y)
            else:
                x1, y1 = x, y
            meade.moveTo(x1, y1, preload=preload)
            
            for r_i in range(nramp):
                visit = takeRamp('n1', nread=nread)
                
                res.append([visit, x1, y1])
                
            if call is not None:
                ret = call(meade)
                callRet.append(ret)

    if led is not None:
        meade.ledsOff()
        
    return pd.DataFrame(res, columns=('visit', 'xsteps', 'ysteps')), callRet

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
    
        tscan = nirander.motorScan(meade, farx, fary, led=(1085,30))
        far.append(tscan)
        tscan = nirander.motorScan(meade, xx, y0, led=(1085,30))
        xrepeats.append(tscan)
    
        tscan = nirander.motorScan(meade, farx, fary, led=(1085,30))
        far.append(tscan)
        tscan = nirander.motorScan(meade, x0, yy, led=(1085,30))
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
        xscan = nirander.motorScan(meade, x0, yy, led=(1085,30), posInPixels=False)
        xrepeats.append(xscan)

        yscan = nirander.motorScan(meade, xx, y0, led=(1085,30), posInPixels=False)
        yrepeats.append(yscan)
    
    xreps = pd.concat(xrepeats, ignore_index=True)
    measureSet(xreps, hxCalib)

    yreps = pd.concat(yrepeats, ignore_index=True)
    measureSet(yreps, hxCalib)
    
    return xreps, yreps

def oversample(grid, center, rad=10, oversample=3):
    indim = slice(center-rad, slice+rad+1)
    outdim = oversample*(rad*2 + 1)
    osImage = np.zeros((dim, dim), dtype=np.float32)

def ditherAt(meade, center, npos=3, nread=3, xsteps=5, ysteps=2):
    if npos%2 != 1:
        raise ValuError("not willing to deal with non-odd dithering")
    rad = npos//2
    xc, yc = meade.pixToSteps(*center)
    x0, y0 = xc-rad, yc-rad
    
    xx = np.arange(x0,x0+xsteps*npos,xsteps)
    yy = np.arange(y0,y0+ysteps*npos,ysteps)

    gridVisits = nirander.motorScan(meade, xx, yy, led=(1085,30), posInPixels=False)

    return gridVisits, oversample(gridVisits)

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
            spotsFrame = spotsFrame.iloc[0].copy()
            spotsFrame[["x","y","x2","y2","peak","flux"]] = np.nan

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
    x = sweep.sort_values(by='focus')['focus']
    y = sweep.sort_values(by='focus')['size']
    poly = np.polynomial.Polynomial.fit(x,y,2)
    c,b,a = poly.convert().coef
    minx = -b/(2*a)
    return minx, poly

def getFocusGrid(seed, spacing=2, r=5):
    focusReq = seed + (np.arange(2*r-1) - (r-1))*spacing
    return focusReq

def _scanForFocus(seed, spacing, r, nread=3, cam='n1'):
    focusReq = getFocusGrid(seed, spacing=spacing, r=r)
    print(focusReq)
    
    if focusReq[0] < 15:
        raise RuntimeError(f"focusReq[0] too low, not starting below: focusReq")
        
    pfsutils.oneCmd(f'xcu_{cam}', f"motors move piston={focusReq[0]-10} abs microns")

    visits = []
    for f in focusReq:
        pfsutils.oneCmd(f'xcu_{cam}', f"motors move piston={f} abs microns")
        visit = takeRamp(cam=cam, nread=nread)
        visits.append(visit)
        
    return pd.DataFrame(dict(visit=visits, focus=focusReq))
               
def scanForFocus(seed, spacing=5, r=4):
    return _scanForFocus(seed, spacing=spacing, r=r)
def scanForCrudeFocus(seed, spacing=25, r=3):
    return _scanForFocus(seed, spacing=spacing, r=r)

def measureSet(scans, hxWork, thresh=1000, center=None, radius=100):
    for f in 'x2', 'y2', 'xpix', 'ypix', 'flux', 'peak':
        scans[f] = np.nan

    for i in range(len(scans)):
        corrImg, spots = getPeaks(hxWork.isr(scans.visit[i]),
                                  center=center, radius=radius,
                                  thresh=thresh, 
                                  mask=hxWork.badMask)
        bestSpot = spots.loc[spots.flux.idxmax()]
        scan_i =  scans.index[i]
        scans.loc[scan_i, 'xpix'] = bestSpot.x
        scans.loc[scan_i, 'ypix'] = bestSpot.y
        scans.loc[scan_i, 'x2'] = bestSpot.x2
        scans.loc[scan_i, 'y2'] = bestSpot.y2
        scans.loc[scan_i, 'size'] =  (bestSpot.x2 + bestSpot.y2)/2
        scans.loc[scan_i, 'flux'] = bestSpot.flux
        scans.loc[scan_i, 'peak'] = bestSpot.peak
    
    return scans
