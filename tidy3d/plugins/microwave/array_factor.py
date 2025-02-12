"""Convenience functions for estimating antenna radiation by applying array factor."""

from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
import pydantic.v1 as pd
from pydantic.v1 import NonNegativeFloat, PositiveInt

from ...components.base import Tidy3dBaseModel
from ...components.data.monitor_data import AbstractFieldProjectionData, DirectivityData
from ...components.data.sim_data import SimulationData
from ...components.geometry.base import Box
from ...components.monitor import AbstractFieldProjectionMonitor
from ...components.simulation import Simulation
from ...components.structure import Structure
from ...components.types import ArrayLike
from ...constants import C_0, inf
from ...log import log


class AbstractAntennaArrayCalculator(Tidy3dBaseModel, ABC):
    """Abstract base for phased array calculators."""

    @abstractmethod
    def make_antenna_array(
        self, simulation: Simulation, duplicate_sources: bool = True
    ) -> Simulation:
        """
        Converts a single antenna simulation into an antenna array simulation.
        This function identifies the size and position of the single antenna
        in the input simulation and uses this information to compute the dimensions
        of the resulting antenna array simulation. All structures, sources, lumped elements,
        and mesh override structures are duplicated, while monitors are extended in size.
        Only projection monitors are transferred into the resulting simulation.

        Parameters:
        ----------
        simulation : Simulation
            The simulation specification describing a single antenna setup.
        duplicate_sources : bool = True
            Make all antennas in the array active.

        Returns:
        --------
        Simulation
            The simulation specification describing the antenna array.
        """

    @abstractmethod
    def array_factor(
        self,
        theta: Union[float, ArrayLike],
        phi: Union[float, ArrayLike],
        frequency: Union[NonNegativeFloat, ArrayLike],
    ) -> ArrayLike:
        """
        Compute the array factor for an antenna array.

        Parameters:
        -----------
        theta : Union[float, ArrayLike]
            Observation angles in the elevation plane (in radians).
        phi : Union[float, ArrayLike]
            Observation angles in the azimuth plane (in radians).
        frequency : Union[NonNegativeFloat, ArrayLike]
            Signal frequency (in Hz).

        Returns:
        --------
        ArrayLike
            Array factor values for each combination of theta and phi.
        """

    def monitor_data_from_array_factor(
        self,
        monitor_data: AbstractFieldProjectionData,
        new_monitor: AbstractFieldProjectionMonitor = None,
    ) -> AbstractFieldProjectionData:
        """Apply the array factor to the monitor data of a single antenna.

        Parameters:
        ----------
        monitor_data : AbstractFieldProjectionData
            The monitor data of a single antenna.
        new_monitor : AbstractFieldProjectionMonitor = None
            The new monitor to be used in the resulting data.

        Returns:
        --------
        AbstractFieldProjectionData
            The monitor data of the antenna array.
        """

        # Get spherical coordinates
        r, theta, phi = list(monitor_data.coords_spherical.values())
        freqs = monitor_data.f
        coords_shape = list(np.shape(r))
        coords_shape.append(len(freqs))

        # Compute the array factor
        af = self.array_factor(theta.ravel(), phi.ravel(), freqs)
        af = np.reshape(af, coords_shape)

        update_dict = {}

        # Apply the array factor to the monitor data
        for key, field in monitor_data.field_components.items():
            update_dict[key] = field.copy() * af

        if new_monitor is not None:
            update_dict["monitor"] = new_monitor

        if isinstance(monitor_data, DirectivityData):
            # Attempt to recompute flux

            # check that theta is defined from 0 to pi
            theta = monitor_data.coords["theta"].data
            good_theta_samplig = np.isclose(np.min(theta), 0) and np.isclose(np.max(theta), np.pi)

            if not good_theta_samplig:
                log.warning("'theta' must sample interval from 0 to pi.")

            # check that phi is defined from 0 to 2 * pi
            phi = monitor_data.coords["phi"].data
            good_phi_samplig = np.isclose(np.min(phi), 0) and np.isclose(np.max(phi), 2 * np.pi)

            if not good_phi_samplig:
                log.warning("'phi' must sample interval from 0 to 2 * pi.")

            s = 0.5 * np.real(
                monitor_data.Etheta * np.conj(monitor_data.Hphi)
                - monitor_data.Ephi * np.conj(monitor_data.Htheta)
            )
            s_sin_theta = s * np.sin(s.theta) * af * np.conj(af)
            P = s_sin_theta.integrate(coord=("theta", "phi")) * s.r * s.r

            update_dict["flux"] = P.isel(r=0)

        else:
            update_dict["projection_surfaces"] = new_monitor.projection_surfaces

        # Create a new monitor data with the updated fields
        new_mnt_data = monitor_data.updated_copy(
            **update_dict,
        )

        return new_mnt_data

    def simulation_data_from_array_factor(
        self,
        antenna_data: SimulationData,
    ) -> SimulationData:
        """
        Computes the far-field data of a rectangular antenna array based on the far-field data of
        a single antenna. Note that any near-field monitor data will be ignored.

        Parameters:
        ----------
        antenna_data : SimulationData
            The far-field data of a single antenna.

        Returns:
        --------
        SimulationData
            The far-field data of the antenna array.
        """

        # create an expanded simulation for reference
        sim_array = self.make_antenna_array(antenna_data.simulation)

        # names of transferred monitors
        mnt_dict = {mnt.name: mnt for mnt in sim_array.monitors}

        # process far field data
        data_array = []
        for mnt_data in antenna_data.data:
            mnt_name = mnt_data.monitor.name
            if mnt_name in mnt_dict:
                array_mnt_data = self.monitor_data_from_array_factor(
                    monitor_data=mnt_data,
                    new_monitor=mnt_dict[mnt_name],
                )
                data_array.append(array_mnt_data)

        return SimulationData(simulation=sim_array, data=data_array)


class RectangularAntennaArrayCalculator(AbstractAntennaArrayCalculator):
    """Convenience calculator for rectangular phased antenna arrays."""

    array_size: Tuple[PositiveInt, PositiveInt, PositiveInt] = pd.Field(
        title="Array Size",
        description="Number of antennas along x, y, and z directions.",
    )

    spacings: Tuple[NonNegativeFloat, NonNegativeFloat, NonNegativeFloat] = pd.Field(
        title="Antenna Spacings",
        description="Center-to-center spacings between antennas along x, y, and z directions.",
    )

    phase_shifts: Tuple[float, float, float] = pd.Field(
        (0, 0, 0),
        title="Phase Shifts",
        description="Phase-shifts between antennas along x, y, and z directions.",
    )

    def make_antenna_array(self, simulation: Simulation):
        """
        Converts a single antenna simulation into an antenna array simulation.
        This function identifies the size and position of the single antenna
        in the input simulation and uses this information to compute the dimensions
        of the resulting antenna array simulation. All structures, sources, lumped elements,
        and mesh override structures are duplicated, while monitors are extended in size.
        Only projection monitors are transferred into the resulting simulation.

        Parameters:
        ----------
        simulation : Simulation
            The simulation specification describing a single antenna setup.
        array_size : Tuple[int, int, int]
            The number of antennas in the x, y, and z directions.
        spacings : Tuple[float, float, float]
            The spacing between antennas in the x, y, and z directions (in micrometers).
        phase_shifts : Tuple[float, float, float]
            The phase shift between antennas in the x, y, and z directions (in radians).

        Returns:
        --------
        Simulation
            The simulation specification describing the antenna array.
        """

        # directions in which we will need to tile simulation
        extend_dims = [dim for dim, size in enumerate(self.array_size) if size > 1]

        # detect bounding box of all structures, sources, and lumped elements in the simulation
        sim_bounds = np.array(simulation.bounds)
        antenna_bounds = sim_bounds.copy()
        for dim in extend_dims:
            antenna_bounds[0][dim] = inf
            antenna_bounds[1][dim] = -inf

        all_objects = list(simulation.structures)
        all_objects += list(simulation.sources)
        all_objects += list(simulation.lumped_elements)

        for obj in all_objects:
            # get bounding box of the object
            if isinstance(obj, Structure):
                obj_bounds = np.array(obj.geometry.bounds)
            else:
                obj_bounds = np.array(obj.bounds)

            for dim in extend_dims:
                # update minimum and maximum bounds in each dimension
                # check if object extends beyond the simulation bounds on both sides
                extends_beyond_min = obj_bounds[0][dim] < sim_bounds[0][dim]
                extends_beyond_max = obj_bounds[1][dim] > sim_bounds[1][dim]
                if extends_beyond_min or extends_beyond_max:
                    print(obj_bounds, sim_bounds)
                    # in case of a box, we just ignore it, since we will be able to extend it later
                    if (
                        isinstance(obj, Structure)
                        and isinstance(obj.geometry, Box)
                        and extends_beyond_min
                        and extends_beyond_max
                    ):
                        continue
                    # otherwise shrink the object bounds to simulation bounds
                    obj_bounds[0][dim] = max(obj_bounds[0][dim], sim_bounds[0][dim])
                    obj_bounds[1][dim] = min(obj_bounds[1][dim], sim_bounds[1][dim])
                    # and show a warning
                    log.warning(
                        f"Object {obj.name} (type: {obj.type}) extends beyond simulation bounds along"
                        f" '{'xyz'[dim]}' axis. Please check your antenna setup for correctness."
                    )
                # update minimum and maximum bounds in each dimension
                antenna_bounds[0][dim] = min(antenna_bounds[0][dim], obj_bounds[0][dim])
                antenna_bounds[1][dim] = max(antenna_bounds[1][dim], obj_bounds[1][dim])

        # compute the center and size of the antenna
        antenna_size = antenna_bounds[1] - antenna_bounds[0]

        # compute buffer between the antenna and simulation boundson each side
        buffer_min = antenna_bounds[0] - sim_bounds[0]
        buffer_max = sim_bounds[1] - antenna_bounds[1]

        # warn if provided spacings are smaller than the size of the antenna
        for dim, spacing in zip(extend_dims, self.spacings):
            if spacing < antenna_size[dim]:
                log.warning(
                    f"Spacing in {'xyz'[dim]} direction is smaller than the detected size of the antenna. "
                    "Please check your antenna setup for correctness."
                )

        # compute total size of the antenna array in x, y, and z directions
        antenna_array_size = (np.array(self.array_size) - 1) * np.array(
            self.spacings
        ) + antenna_size

        # compute the total size of the simulation domain
        sim_size = antenna_array_size + buffer_min + buffer_max

        # duplicate structures, sources, lumped elements, and override structures
        array_structures = []
        array_sources = []
        array_lumped_elements = []
        array_overrides = []

        for i in range(self.array_size[0]):
            for j in range(self.array_size[1]):
                for k in range(self.array_size[2]):
                    # compute displacement of the structure in x, y, and z directions
                    translation_vector = (
                        -sim_size / 2
                        - sim_bounds[0]
                        + np.array([i, j, k]) * np.array(self.spacings)
                    )
                    for structure in simulation.structures:

                    	# TODO: expand inf boxes instead of duplicating
                        # create a copy of the original structure by translating it in x, y, and z directions
                        new_structure = structure.updated_copy(
                            geometry=structure.geometry.translated(*translation_vector)
                        )
                        # add the new structure to the list of structures
                        array_structures.append(new_structure)

                    for source in simulation.sources:
                        # create a copy of the original source by translating it in x, y, and z directions
                        phase_diff = (
                            i * self.phase_shifts[0]
                            + j * self.phase_shifts[1]
                            + k * self.phase_shifts[2]
                        )
                        new_source = source.updated_copy(
                            center=tuple(source.center + translation_vector),
                            name=f"{source.name}_{i}_{j}_{k}",
                            source_time=source.source_time.updated_copy(
                                phase=source.source_time.phase + phase_diff
                            ),
                        )
                        # add the new source to the list of sources
                        array_sources.append(new_source)

                    for lumped_element in simulation.lumped_elements:
                        # create a copy of the original lumped element by translating it in x, y, and z directions
                        new_lumped_element = lumped_element.updated_copy(
                            center=tuple(lumped_element.center + translation_vector),
                            name=f"{lumped_element.name}_{i}_{j}_{k}",
                        )
                        # add the new lumped element to the list of lumped elements
                        array_lumped_elements.append(new_lumped_element)

                    for override in simulation.grid_spec.override_structures:
                        # create a copy of the original override by translating it in x, y, and z directions
                        new_override = override.updated_copy(
                            geometry=override.geometry.translated(*translation_vector).bounding_box
                        )
                        # add the new override to the list of overrides
                        array_overrides.append(new_override)

        # expand far-field monitors
        array_monitors = []
        for monitor in simulation.monitors:
            # we only expand field projection monitors
            if isinstance(monitor, AbstractFieldProjectionMonitor):
                # get original monitor bounds
                mnt_bounds = np.array(monitor.bounds)
                mnt_size = np.array(monitor.size)
                mnt_center = np.array(monitor.center)

                if any(mnt_size[dim] == 0 for dim in extend_dims):
                    log.warning(
                        f"Monitor '{monitor.name}' (type: '{monitor.type}') has zero size along "
                        f"{'xyz'[dim]} axis. It will not be included in the resulting simulation."
                    )
                    continue

                for dim in extend_dims:
                    if mnt_size[dim] != np.inf:
                        # check that monitor covers the estimated antenna box
                        if (
                            mnt_bounds[0][dim] > antenna_bounds[0][dim]
                            or mnt_bounds[1][dim] < antenna_bounds[1][dim]
                        ):
                            log.warning(
                                f"Monitor '{monitor.name}' (type: '{monitor.type}') does not cover "
                                f"the estimated antenna box along '{'xyz'[dim]}' axis. "
                                "The automatically extended monitor will likely be wrong. "
                                "Please double check the resulting simulation."
                            )
                        # shift min bounds to new location
                        mnt_bounds[0][dim] = (
                            mnt_bounds[0][dim] - sim_bounds[0][dim] - sim_size[dim] / 2
                        )

                        # shift max bounds to new location
                        mnt_bounds[1][dim] = (
                            mnt_bounds[0][dim]
                            + monitor.size[dim]
                            + (self.array_size[dim] - 1) * self.spacings[dim]
                        )

                        # calculate new monitor size and center
                        mnt_size[dim] = mnt_bounds[1][dim] - mnt_bounds[0][dim]
                        mnt_center[dim] = 0.5 * (mnt_bounds[0][dim] + mnt_bounds[1][dim])

                # create a copy of the original monitor with updated size and center
                new_mnt = monitor.updated_copy(
                    center=tuple(mnt_center),
                    size=tuple(mnt_size),
                )

                # add the new monitor to the list of monitors
                array_monitors.append(new_mnt)

            # otherwise we ignore and warn the user
            else:
                log.warning(
                    f"Monitor '{monitor.name}' (type: '{monitor.type}') will not be automatically "
                    "transferred into the resulting antenna array simulation."
                )

        new_sim = simulation.updated_copy(
            center=(0, 0, 0),
            size=tuple(sim_size),
            structures=array_structures,
            monitors=array_monitors,
            sources=array_sources,
            lumped_elements=array_lumped_elements,
            grid_spec=simulation.grid_spec.updated_copy(override_structures=array_overrides),
        )

        return new_sim

    def array_factor(
        self,
        theta: Union[float, ArrayLike],
        phi: Union[float, ArrayLike],
        frequency: Union[NonNegativeFloat, ArrayLike],
    ) -> ArrayLike:
        """
        Compute the array factor for a 3D antenna array.

        Parameters:
        -----------
        theta : Union[float, ArrayLike]
            Observation angles in the elevation plane (in radians).
        phi : Union[float, ArrayLike]
            Observation angles in the azimuth plane (in radians).
        frequency : Union[NonNegativeFloat, ArrayLike]
            Signal frequency (in Hz).

        Returns:
        --------
        ArrayLike
            Array factor values for each combination of theta and phi.
        """

        # Convert all inputs to numpy arrays
        theta_array = np.atleast_1d(theta)
        phi_array = np.atleast_1d(phi)

        # ensure that theta and phi have the same length
        if len(theta_array) != len(phi_array):
            raise ValueError("'theta' and 'phi' must have the same length")

        # reshape inputs for easier broadcasting
        theta_array = np.reshape(theta_array, (len(theta_array), 1))
        phi_array = np.reshape(phi_array, (len(phi_array), 1))
        f_array = np.reshape(frequency, (1, len(np.atleast_1d(frequency))))

        wavelength = C_0 / f_array
        k = 2 * np.pi / wavelength  # Wavenumber

        # Calculate the phase shift in the x, y, and z directions
        psi_x = (
            k * self.spacings[0] * np.sin(theta_array) * np.cos(phi_array) - self.phase_shifts[0]
        )
        psi_y = (
            k * self.spacings[1] * np.sin(theta_array) * np.sin(phi_array) - self.phase_shifts[1]
        )
        psi_z = k * self.spacings[2] * np.cos(theta_array) - self.phase_shifts[2]

        # Calculate the array factor in the x, y, and z directions
        af_x = np.sum(
            np.exp(-1j * np.arange(self.array_size[0])[:, None, None] * psi_x[np.newaxis, :]),
            axis=0,
        )
        af_y = np.sum(
            np.exp(-1j * np.arange(self.array_size[1])[:, None, None] * psi_y[np.newaxis, :]),
            axis=0,
        )
        af_z = np.sum(
            np.exp(-1j * np.arange(self.array_size[2])[:, None, None] * psi_z[np.newaxis, :]),
            axis=0,
        )

        # Calculate the overall array factor
        array_factor = af_x * af_y * af_z

        return array_factor
