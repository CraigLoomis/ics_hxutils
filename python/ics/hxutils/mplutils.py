import matplotlib as mpl
import numpy as np

from matplotlib.colors import Normalize

class CenteredNorm(Normalize):
    def __init__(self, vcenter=0, halfrange=None, clip=False):
        """
        Normalize symmetrical data around a center (0 by default).
        Unlike `TwoSlopeNorm`, `CenteredNorm` applies an equal rate of change
        around the center.
        Useful when mapping symmetrical data around a conceptual center
        e.g., data that range from -2 to 4, with 0 as the midpoint, and
        with equal rates of change around that midpoint.
        Parameters
        ----------
        vcenter : float, default: 0
            The data value that defines ``0.5`` in the normalization.
        halfrange : float, optional
            The range of data values that defines a range of ``0.5`` in the
            normalization, so that *vcenter* - *halfrange* is ``0.0`` and
            *vcenter* + *halfrange* is ``1.0`` in the normalization.
            Defaults to the largest absolute difference to *vcenter* for
            the values in the dataset.
        Examples
        --------
        This maps data values -2 to 0.25, 0 to 0.5, and 4 to 1.0
        (assuming equal rates of change above and below 0.0):
            >>> import matplotlib.colors as mcolors
            >>> norm = mcolors.CenteredNorm(halfrange=4.0)
            >>> data = [-2., 0., 4.]
            >>> norm(data)
            array([0.25, 0.5 , 1.  ])
        """
        super().__init__(vmin=None, vmax=None, clip=clip)
        self._vcenter = vcenter
        # calling the halfrange setter to set vmin and vmax
        self.halfrange = halfrange

    def _set_vmin_vmax(self):
        """
        Set *vmin* and *vmax* based on *vcenter* and *halfrange*.
        """
        self.vmax = self._vcenter + self._halfrange
        self.vmin = self._vcenter - self._halfrange

    def autoscale(self, A):
        """
        Set *halfrange* to ``max(abs(A-vcenter))``, then set *vmin* and *vmax*.
        """
        A = np.asanyarray(A)
        self._halfrange = max(self._vcenter-A.min(),
                              A.max()-self._vcenter)
        self._set_vmin_vmax()

    def autoscale_None(self, A):
        """Set *vmin* and *vmax*."""
        A = np.asanyarray(A)
        if self._halfrange is None and A.size:
            self.autoscale(A)

    @property
    def vcenter(self):
        return self._vcenter

    @vcenter.setter
    def vcenter(self, vcenter):
        if vcenter != self._vcenter:
            self._vcenter = vcenter
            self._changed()
        if self.vmax is not None:
            # recompute halfrange assuming vmin and vmax represent
            # min and max of data
            self._halfrange = max(self._vcenter-self.vmin,
                                  self.vmax-self._vcenter)
            self._set_vmin_vmax()

    @property
    def halfrange(self):
        return self._halfrange

    @halfrange.setter
    def halfrange(self, halfrange):
        if halfrange is None:
            self._halfrange = None
            self.vmin = None
            self.vmax = None
        else:
            self._halfrange = abs(halfrange)

    def __call__(self, value, clip=None):
        if self._halfrange is not None:
            # enforce symmetry, reset vmin and vmax
            self._set_vmin_vmax()
        return super().__call__(value, clip=clip)
