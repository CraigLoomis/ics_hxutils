import fitsio
import numpy as np
import pathlib

class HxRamp(object):
    def __init__(self, fitsOrPath):
        """Basic H4 ramp object

        Args
        ----
        fitsOrPath : `fitsio.FITS` or path-like.
            The path to open as a FITS file or an existing FITS oject to use.
        """
        if not isinstance(fitsOrPath, fitsio.FITS):
            fitsOrPath = fitsio.FITS(fitsOrPath)
        self.fits = fitsOrPath

    def __str__(self):
        return f"HxRamp(filename={self.fits._filename}, nReads={self.nreads})"

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        """ fitsio caches open files, so try to close when we know we want to."""
        self.fits.close()
        return True

    @property
    def nreads(self):
        return (len(self.fits)-1)//2

    def _readIdxToAbsoluteIdx(self, n):
        """Convert possibly negative ramp read index into positive one"""
        nums = range(self.nreads)
        return nums[n]

    def dataN(self, n):
        """Return the data plane for the n-th read.

        Args
        ----
        n : `int`
          0-indexed read number

        Returns
        -------
        im : np.uint16 image
        """
        n = self._readIdxToAbsoluteIdx(n)
        extname = f'IMAGE_{n+1}'
        return self.fits[extname].read()

    def irpN(self, n):
        """Return the reference plane for the n-th read.

        If the IRP HDU is empty we did not acquire using IRP. So return 0.

        Does not interpolate N:1 IRP planes to full size images.

        Args
        ----
        n : `int`
          0-indexed read number

        Returns
        -------
        im : np.uint16 image, or 0 if it is empty
        """
        n = self._readIdxToAbsoluteIdx(n)
        extname = f'IRP_{n+1}'
        irpImage = self.fits[extname].read()
        if irpImage is None or irpImage.shape == (1,1):
            return np.uint16(0)
        else:
            return irpImage

    def readN(self, n):
        """Return the IRP-corrected image for the n-th read.

        Note that data - ref is often negative, so we convert to float32 here.

        Args
        ----
        n : `int`
          0-indexed read number

        Returns
        -------
        im : np.float32 image
        """
        return self.dataN(n).astype('f4') - self.irpN(n).astype('f4')

    def cds(self, r0=0, r1=-1):
        """Return the CDS image between two reads.

        This is the most common way to get an image from an IRP H4 ramp.

        Args
        ----
        r0 : `int`
          0-indexed read number of the 1st read
        r1 : `int`
          0-indexed read number of the 2st read


        Returns
        -------
        im : np.float32 image
        """
        return self.readN(r1) - self.readN(r0)
