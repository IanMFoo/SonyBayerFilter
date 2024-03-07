"""Module for the abstract Filter class and all its implementations."""

import abc
from numbers import Number, Rational

import numpy as np
import numpy.typing as npt
from overrides import override

from vipdopt.utils import sech

SIGMOID_BOUNDS = (0.0, 1.0)


# TODO: design a way for different filters to take different arguments in methods
# TODO: Make code more robust to inputs being arrays iinstead of scalars
class Filter(abc.ABC):
    """An abstract interface for Filters."""

    @abc.abstractproperty
    def _bounds(self):
        pass

    @abc.abstractproperty
    def init_vars(self) -> dict:
        """The variables used to initalize this Filter."""

    def verify_bounds(self, variable: npt.NDArray | Number) -> bool:
        """Checks if variable is within bounds of this filter.

        Variable can either be a single number or an array of numbers.
        """
        return bool(
            (np.min(np.array(variable)) >= self._bounds[0])
            and (np.max(np.array(variable)) <= self._bounds[1])
        )

    @abc.abstractmethod
    def forward(self, x: npt.NDArray | Number) -> npt.NDArray | Number:
        """Propogate x through the filter and return the result."""

    @abc.abstractmethod
    def fabricate(self, x: npt.NDArray | Number) -> npt.NDArray | Number:
        """Propogate x through the filter and return the result, binarizing values."""

    @abc.abstractmethod
    def chain_rule(
        self,
        deriv_out: npt.NDArray | Number,
        var_out: npt.NDArray | Number,
        var_in: npt.NDArray | Number,
    ) -> npt.NDArray | Number:
        """Apply the chain rule and propagate the derivative back one step."""

class Sigmoid(Filter):
    """Applies a sigmoidal projection filter to binarize an input.

    Takes an input auxiliary density p(x) ranging from 0 to 1 and applies a sigmoidal
    projection filter to binarize / push it to either extreme. This depends on the
    strength of the filter. See OPTICA paper supplement Section IIA,
    https://doi.org/10.1364/OPTICA.384228,  for details. See also Eq. (9) of
    https://doi.org/10.1007/s00158-010-0602-y.

    Attributes:
        eta (Number): The center point of the sigmoid. Must be in range [0, 1].
        beta (Number): The strength of the sigmoid
        _bounds (tuple[Number]): The bounds of the filter. Always equal to (0, 1)
        _denominator (Number | npt.NDArray): The denominator used in various methods;
            for reducing re-computation.
    """
    @property
    @override
    def _bounds(self):
        return SIGMOID_BOUNDS

    @property
    @override
    def init_vars(self) -> dict:
        return {'eta': self.eta, 'beta': self.beta}

    def __init__(self, eta: Rational, beta: Rational) -> None:
        """Initialize a sigmoid filter based on eta and beta values."""
        if not self.verify_bounds(eta):
            raise ValueError('Eta must be in the range [0, 1]')

        self.eta  = eta
        self.beta = beta

        # Calculate denominator for use in methods
        self._denominator = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))

    def __repr__(self) -> str:
        """Return a string representation of the filter."""
        return f'Sigmoid filter with eta={self.eta:0.3f} and beta={self.beta:0.3f}'

    def __eq__(self, __value: object) -> bool:
        """Test equality."""
        if isinstance(__value, Sigmoid):
            return self.eta == __value.eta and self.beta == __value.beta
        return super().__eq__(__value)

    @override
    def forward(self, x: npt.NDArray | Number) -> npt.NDArray | Number:
        """Propogate x through the filter and return the result.
        All input values of x above the threshold eta, are projected to 1, and the
        values below, projected to 0. This is Eq. (9) of https://doi.org/10.1007/s00158-010-0602-y.
        """
        numerator = np.tanh(self.beta * self.eta) + \
            np.tanh(self.beta * (np.copy(np.array(x)) - self.eta))
        return numerator / self._denominator

    @override  # type: ignore
    def chain_rule(
        self,
        deriv_out: npt.NDArray | Number,
        var_out: npt.NDArray | Number,
        var_in: npt.NDArray | Number,
    ) -> npt.NDArray | Number:
        """Apply the chain rule and propogate the derivative back one step.

        Returns the first argument, multiplied by the direct derivative of forward()
        i.e. Eq. (9) of https://doi.org/10.1007/s00158-010-0602-y, with respect to
        \\tilde{p_i}.
        """
        del var_out  # not needed for sigmoid filter

        numerator = self.beta * \
            np.power(sech(self.beta * (var_in - self.eta)), 2) # type: ignore
        return deriv_out * numerator / self._denominator            #! 20240228 Ian - Fixed, was missing a deriv_out factor

    @override
    def fabricate(self, x: npt.NDArray | Number) -> npt.NDArray | Number:
        """Apply filter to input as a hard step-function instead of sigmoid.

        Returns:
            _bounds[0] where x <= eta, and _bounds[1] otherwise
        """
        fab = np.array(x)
        fab[fab <= self.eta] = self._bounds[0]
        fab[fab > self.eta] = self._bounds[1]
        if isinstance(x, Number):  # type: ignore
            return fab.item()
        return fab

class Scale(Filter):
    """Assuming that there is an input between 0 and 1, scales it to the range, min, and max, that are declared during initialization.
    See OPTICA paper supplement Section IIA, https://doi.org/10.1364/OPTICA.384228,  for details.
    This class is used directly after the sigmoid filter is applied.

    Attributes:
        variable_bounds: here they will denote the permittivity bounds
    """

    @property
    @override
    def _bounds(self):
        return self.variable_bounds

    @property
    @override
    def init_vars(self) -> dict:
        return {'variable_bounds': self.variable_bounds}

    def __init__(self, variable_bounds):
        self.variable_bounds = variable_bounds

        #
        # variable_bounds specifies the minimum and maximum value of the scaled variable.
        # We assume we are scaling an input that is between 0 and 1 -> between min_permittivity and max_permittivity
        #

        self.min_value = variable_bounds[0]
        self.max_value = variable_bounds[1]
        self.range = self.max_value - self.min_value

    def __repr__(self):
        """Return a string representation of the filter."""
        return f'Scale filter with minimum={self.min_value:0.3f} and max={self.max_value:0.3f}'

    # @override
    def forward(self, x):
        """Performs scaling according to min and max values declared. Assumed that input is between 0 and 1."""
        return np.add(self.min_value, np.multiply(self.range, x))

    # @override  # type: ignore
    def chain_rule(
        self,
        deriv_out,
        var_out,
        var_in	):
        """Apply the chain rule and propagate the derivative back one step."""
        return np.multiply(self.range, deriv_out)

    # @override
    def fabricate(self, x):
        """Calls forward()."""
        return self.forward(x)
