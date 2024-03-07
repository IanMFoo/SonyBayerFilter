"""Class for general sources in a simulation."""


import numpy.typing as npt

from vipdopt.simulation import LumericalSimulation


class Monitor:
    """Class representing the different source monitors in a simulation."""

    def __init__(
            self,
            sim: LumericalSimulation,
            source_name: str,
            monitor_name: str,
    ) -> None:
        """Initialize a Monitor."""
        self.sim = sim
        self.source_name = source_name
        self.monitor_name = monitor_name

        # Initialize field values
        self._tshape = None  # transmission shape
        self._fshape = None  # field shape
        self._e = None
        self._h = None
        self._trans_mag = None

    def __eq__(self, __value: object) -> bool:
        """Test equality."""
        if isinstance(__value, Monitor):
            return self.sim == __value.sim and self.monitor_name == __value.monitor_name
        return None

    @property
    def tshape(self) -> tuple[int, ...]:
        """Return the shape of the numpy array for this monitor's fields."""
        if self._tshape is None and self.sim.fdtd is not None:
            self._tshape = self.sim.get_transmission_shape(self.monitor_name)
        return self._tshape

    @property
    def fshape(self) -> tuple[int, ...]:
        """Return the shape of the numpy array for this monitor's fields."""
        if self._fshape is None and self.sim.fdtd is not None:
            self._fshape = self.e.shape                 #! 20240228 Ian - We need the design E-field shape not the design index shape
        return self._fshape

    @property
    def e(self) -> npt.NDArray:
        """Return the e field produced by this source."""
        if self._e is None:
            self._e = self.sim.get_efield(self.monitor_name)
        return self._e

    @property
    def h(self) -> npt.NDArray:
        """Return the h field produced by this source."""
        if self._h is None:
            self._h = self.sim.get_hfield(self.monitor_name)
        return self._h

    @property
    def trans_mag(self) -> npt.NDArray:
        """Return the transmission magnitude of this source."""
        if self._trans_mag is None:
            self._trans_mag = self.sim.get_transmission_magnitude(self.monitor_name)
        return self._trans_mag
