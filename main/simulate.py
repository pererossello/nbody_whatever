import os
import shutil
import time

import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange, float32, int32, void
import h5py


@jit(
    void(float32[:, :], int32, float32[:, :], float32[:], float32, float32),
    nopython=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def net_fields(fields, N, POS, MASS, e, G):
    """
    Calculates gravitational forces between N bodies, avoiding self-interaction and using pairwise calculations. Optimized with Numba for performance.
    """
    acc = np.zeros((N, 2), dtype=np.float32)
    for i in prange(N):
        acc = acc * 0
        for j in prange(N):
            if i == j:
                continue
            dx = POS[i][0] - POS[j][0]
            dy = POS[i][1] - POS[j][1]
            r2 = dx * dx + dy * dy
            F_over_mm = -G * np.power(r2 + e * e, -1.5)
            Fx = F_over_mm * dx
            Fy = F_over_mm * dy

            acc[j][0] = Fx * MASS[j]
            acc[j][1] = Fy * MASS[j]

        fields[i, 0] = np.sum(acc[:, 0])
        fields[i, 1] = np.sum(acc[:, 1])


@jit(
    void(
        float32[:, :],
        float32[:, :],
        float32[:, :],
        float32[:],
        float32,
        int32,
        int32,
        float32,
        float32,
    ),
    nopython=True,
    fastmath=True,
    parallel=False,
)
def update_positions(fields, POS, VEL, MASS, dt, is_first_step, N, e, G):
    """
    Updates particle positions and velocities using a Leapfrog Step integration method. Considers whether it's the first integration step to precompute the fields.
    """

    if is_first_step == 1:
        net_fields(fields, N, POS, MASS, e, G)

    VEL += 0.5 * fields * dt
    POS += VEL * dt

    net_fields(fields, N, POS, MASS, e, G)
    VEL += 0.5 * fields * dt


class NBodySimulation:
    def __init__(self, path_ics, snap_path):
        """
        Initializes the simulation with paths for initial conditions and snapshot output, and loads initial state data from an HDF5 file.
        """

        self.simulation_done = False
        self.path_ics = path_ics
        self.path_output = snap_path

        self._create_output_file()

        with h5py.File(self.path_ics, "r") as file:
            self.dimensions = file["Header"].attrs["Dimensions"]
            self.N = file["Header"].attrs["N"]

            self.POS = np.array(file["Bodies"]["Positions"])
            self.VEL = np.array(file["Bodies"]["Velocities"])
            self.MASS = np.array(file["Bodies"]["Masses"])

    def set_time(self, duration, time_step, snapshots=100):
        """
        Sets the simulation duration, time step, and the number of snapshots based on the total duration.
        """

        self.duration = np.array(duration, dtype=np.float32)
        self.time_step = np.float32(time_step)

        max_snapshots = int(duration / time_step)

        if snapshots > max_snapshots:
            snapshots = max_snapshots

        self.snap_times = np.linspace(0.0, duration, snapshots)

    def run_simulation(self, e=0.01, G=1):
        """
        Runs the N-body simulation, managing time steps, updating positions, handling first step differentiation, and saving snapshots at predefined times.
        """

        if self.simulation_done:
            raise ValueError("Create new instance to run a new simulation.")

        self.e = np.float32(e)
        self.G = np.float32(G)

        fields = np.zeros((self.N, 2), dtype=np.float32)

        self.time = 0.0
        is_first_step = 1
        self.snapshot_idx = 0
        start_clock = time.time()
        while self.time < self.duration:

            update_positions(
                fields,
                self.POS,
                self.VEL,
                self.MASS,
                self.time_step,
                is_first_step,
                self.N,
                self.e,
                self.G,
            )
            self.time += self.time_step
            is_first_step = 0

            if self.time >= self.snap_times[self.snapshot_idx]:
                self.snapshot_idx += 1
                self._save_snapshot()

            perc = self.time / (self.duration + self.time_step) * 100

            print(f"\r{perc:.2f}%", end="")

        end_clock = time.time()
        print(f"\nElapsed time: {end_clock - start_clock:.2f} s")

        self.simulation_done = True

        self._finish_up_output_file()

    def estimate_runtime(self, num_steps=10):
        """
        Estimates runtime for the simulation by running a smaller number of time steps and extrapolating the total time needed.
        """

        pos_temp = np.random.rand(self.N, self.dimensions).astype(np.float32)
        vel_temp = np.random.rand(self.N, self.dimensions).astype(np.float32)
        mass_temp = np.random.rand(self.N).astype(np.float32)
        fields = np.zeros((self.N, 2), dtype=np.float32)

        step_clocktime = 0
        is_first_step = 1
        step = 0

        start_clock = time.time()
        while step < num_steps:
            start_clock_step = time.time()
            update_positions(
                fields,
                pos_temp,
                vel_temp,
                mass_temp,
                self.time_step,
                is_first_step,
                self.N,
                np.float32(1),
                np.float32(1),
            )
            is_first_step = 0

            end_clock_step = time.time()
            step_clocktime += end_clock_step - start_clock_step
            step += 1

        end_clock = time.time()
        elapsed_time = end_clock - start_clock
        estimated_time = elapsed_time * self.duration / (self.time_step * num_steps)
        hours = int(estimated_time / 3600)
        minutes = int((estimated_time - hours * 3600) / 60)
        seconds = int(estimated_time - hours * 3600 - minutes * 60)

        print(f"Estimated time: {hours:02d}h {minutes:02d}m {seconds:02d}s")
        print(f"Mean time per step: {step_clocktime/num_steps:.2e} s")

    def _save_snapshot(self):
        """
        Saves the current state of the simulation to an output file as a snapshot at the current time step.
        """
        with h5py.File(self.path_output, "a") as f:

            step_group = f.create_group(f"{self.snapshot_idx:04d}")
            step_group.create_dataset("Positions", data=self.POS, dtype=np.float32)
            step_group.create_dataset("Velocities", data=self.VEL, dtype=np.float32)
            step_group.create_dataset("Masses", data=self.MASS, dtype=np.float32)
            step_group.attrs["Time"] = self.time

        return

    def _create_output_file(self):
        """
        Prepares the output file structure based on the initial conditions file, ensuring directories exist and old data is cleared.
        """

        if not os.path.exists(self.path_output):
            save_dir = os.path.dirname(self.path_output)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        else:
            os.remove(self.path_output)

        with h5py.File(self.path_ics, "r") as file_ics:
            with h5py.File(self.path_output, "w") as file_output:

                for group_name in file_ics.keys():

                    if group_name != "Bodies":
                        group = file_output.create_group(group_name)
                    else:
                        step = 0
                        new_name = f"{step:04d}"
                        group = file_output.create_group(new_name)

                    for attr_name, attr_value in file_ics[group_name].attrs.items():
                        group.attrs[attr_name] = attr_value

                    for dataset_name in file_ics[group_name].keys():
                        source_dataset = file_ics[group_name][dataset_name]
                        group.create_dataset(dataset_name, data=source_dataset[...])

    def _finish_up_output_file(self):
        """
        Finalizes the output file by updating the header with the total number of snapshots.
        """
        with h5py.File(self.path_output, "a") as file:
            file["Header"].attrs["NSnapshots"] = len(list(file.keys())) - 2
