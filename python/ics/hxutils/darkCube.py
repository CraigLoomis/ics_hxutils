from importlib import reload

import numpy as np
import fitsio

from ics.hxutils import hxstack as hx
reload(hx)

class DarkCube(object):
    def __init__(self, cube=None, visits=None, center=None):
        self._cube = cube
        self.visits = visits
        self.nread = len(self._cube)
        self.center = center

    @property
    def visit0(self):
        return self.visits[0]
    
    def __str__(self):
        return f"DarkCube(nread={self.nread}, visits={self.visits})"
    def __repr__(self):
        return str(self)
                   
    def __getitem__(self, ii):
        return self.cds(r1=ii)
    
    def dump(self, path):
        visitStr = ' '.join([str(v) for v in self.visits])
        hdr = [dict(name='VISITS', value=visitStr, comment="list of visits in dark"),
               dict(name='NREAD', value=self.nread, comment="number of reads in dark"),
               dict(name='NVISIT', value=len(self.visits), comment="number of visits in dark"),]

        ff = fitsio.FITS(path, mode='rw', clobber=True)
        ff.write(data=None, header=hdr)

        for i in range(self.nread):
            ff.write(self._cube[i]) # , compress='GZIP_2')  # Jesus this takes forever!
        ff.close()

    @classmethod
    def createFromVisits(cls, visits, cam=None):
        cube = hx.medianCubes(visits, cam=cam)

        self = cls(cube, visits=visits)
        return self
        
    @classmethod
    def loadFromFits(cls, path):
        ff = fitsio.FITS(path, mode='r')
        hdr = ff[0].read_header()
        visits = [int(v) for v in hdr['VISITS'].split()]
        
        cds = []
        for i in range(len(visits)):
            cds.append(ff[i+1].read())
        cube = np.stack(cds)

        self = cls(cube, visits=visits)
        return self
                       
    def cds(self, r0=0, r1=-1):
        if r0 != 0:
            raise ValueError(f'can only take full CDS from DarkCubes, not with r0={r0}')
        return self._cube[r1]
    
def load(path):
    return DarkCube.loadFromFits(path)
