"""Storing data associated with results from the TerminalComponentModeler"""

import pydantic.v1 as pd
import xarray as xr

from ....components.data.data_array import (
    DataArray,
    FieldProjectionAngleDataArray,
    FreqDataArray,
)
from ....components.data.monitor_data import (
    DirectivityData,
)
from ....components.types import PolarizationBasis


class PortDataArray(DataArray):
    """Array of values over dimensions of frequency and port name.

    Example
    -------
    >>> import numpy as np
    >>> f = [2e9, 3e9, 4e9]
    >>> ports = ["port1", "port2"]
    >>> coords = dict(f=f, port=ports)
    >>> data = (1+1j) * np.random.random((3, 2))
    >>> pd = PortDataArray(data, coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "port")


class TerminalPortDataArray(DataArray):
    """Port parameter matrix elements for terminal-based ports.

    Example
    -------
    >>> import numpy as np
    >>> ports_in = ["port1", "port2"]
    >>> ports_out = ["port1", "port2"]
    >>> f = [2e14]
    >>> coords = dict(f=f, port_out=ports_out, port_in=ports_in)
    >>> data = (1+1j) * np.random.random((1, 2, 2))
    >>> td = TerminalPortDataArray(data, coords=coords)
    """

    __slots__ = ()
    _dims = ("f", "port_out", "port_in")
    _data_attrs = {"long_name": "terminal-based port matrix element"}


class AntennaParametersData(DirectivityData):
    """Data representing the main parameters and figures of merit for antennas.

    Example
    -------
    >>> import numpy as np
    >>> from tidy3d.components.data.monitor_data import FluxDataArray, FieldProjectionAngleDataArray
    >>> from tidy3d.components.monitor import DirectivityMonitor
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> r = np.atleast_1d(1e6)
    >>> theta = np.linspace(0, np.pi, 10)
    >>> phi = np.linspace(0, 2*np.pi, 20)
    >>> coords = dict(r=r, theta=theta, phi=phi, f=f)
    >>> coords_flux = dict(f=f)
    >>> field_values = (1+1j) * np.random.random((len(r), len(theta), len(phi), len(f)))
    >>> flux_data = FluxDataArray(np.random.random(len(f)), coords=coords_flux)
    >>> scalar_field = FieldProjectionAngleDataArray(field_values, coords=coords)
    >>> monitor = DirectivityMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,2),
    ...     freqs=f,
    ...     name="rad_monitor",
    ...     phi=phi,
    ...     theta=theta
    ... )
    >>> power_data = FreqDataArray(np.random.random(len(f)), coords=coords_flux)
    >>> data = AntennaParametersData(
    ...     monitor=monitor,
    ...     projection_surfaces=monitor.projection_surfaces,
    ...     flux=flux_data,
    ...     Er=scalar_field,
    ...     Etheta=scalar_field,
    ...     Ephi=scalar_field,
    ...     Hr=scalar_field,
    ...     Htheta=scalar_field,
    ...     Hphi=scalar_field,
    ...     power_incident=power_data,
    ...     power_reflected=power_data
    ... )

    Notes
    -----
    The definitions of radiation efficiency, reflection efficiency, gain, and realized gain
    are based on:

    Balanis, Constantine A., "Antenna Theory: Analysis and Design,"
    John Wiley & Sons, Chapter 2.9 (2016).
    """

    power_incident: FreqDataArray = pd.Field(
        ...,
        title="Power incident",
        description="Array of values representing the incident power to an antenna.",
    )

    power_reflected: FreqDataArray = pd.Field(
        ...,
        title="Power reflected",
        description="Array of values representing power reflected due to an impedance mismatch with the antenna.",
    )

    @staticmethod
    def from_directivity_data(
        dir_data: DirectivityData, power_inc: FreqDataArray, power_refl: FreqDataArray
    ):
        """Convenience method for creating :class:.`AntennaParametersData` directly
        from :class:.`DirectivityData`, as well as, the power incident to and power
        reflected from the antenna.
        """
        antenna_params_dict = {
            **dir_data.dict(),
            "power_incident": power_inc,
            "power_reflected": power_refl,
        }
        antenna_params_dict.pop("type")
        return AntennaParametersData(**antenna_params_dict)

    @property
    def radiation_efficiency(self) -> FreqDataArray:
        """The radiation efficiency of the antenna."""
        return self.calc_radiation_efficiency(self.power_incident - self.power_reflected)

    @property
    def reflection_efficiency(self) -> FreqDataArray:
        """The reflection efficiency of the antenna, which is due to an impedance mismatch."""
        total_power = self.power_incident + self.power_reflected
        reflection_efficiency = self.power_incident / total_power
        return reflection_efficiency

    def partial_gain(self, pol_basis: PolarizationBasis = "linear") -> xr.Dataset:
        """The partial gain figures of merit for antennas. The partial gains are computed
        in the ``linear`` or ``circular`` polarization bases. Gain is dimensionless.

        Parameters
        ----------
        pol_basis : PolarizationBasis
            The desired polarization basis used to express partial gain, either
            ``linear`` or ``circular``.

        Returns
        -------
        ``xarray.Dataset``
            Dataset containing the partial gains split into the two polarization states.
        """
        self._check_valid_pol_basis(pol_basis)
        partial_D = self.partial_directivity(pol_basis=pol_basis)
        if pol_basis == "linear":
            rename_mapping = {"Dtheta": "Gtheta", "Dphi": "Gphi"}
        else:
            rename_mapping = {"Dright": "Gright", "Dleft": "Gleft"}
        return self.radiation_efficiency * partial_D.rename(rename_mapping)

    @property
    def gain(self) -> FieldProjectionAngleDataArray:
        """The gain figure of merit for antennas. Gain is dimensionless."""
        partial_G = self.partial_gain()
        return partial_G.Gtheta + partial_G.Gphi

    def partial_realized_gain(self, pol_basis: PolarizationBasis = "linear") -> xr.Dataset:
        """The partial realized gain figures of merit for antennas. The partial gains are computed
        in the ``linear`` or ``circular`` polarization bases. Gain is dimensionless.

        Parameters
        ----------
        pol_basis : PolarizationBasis
            The desired polarization basis used to express partial gain, either
            ``linear`` or ``circular``.

        Returns
        -------
        ``xarray.Dataset``
            Dataset containing the partial realized gains split into the two polarization states.
        """
        self._check_valid_pol_basis(pol_basis)
        reflection_efficiency = self.reflection_efficiency
        partial_G = self.partial_gain(pol_basis=pol_basis)
        return reflection_efficiency * partial_G

    @property
    def realized_gain(self) -> FieldProjectionAngleDataArray:
        """The realized gain figure of merit for antennas. Realized gain is dimensionless."""
        partial_G = self.partial_realized_gain()
        return partial_G.Gtheta + partial_G.Gphi
