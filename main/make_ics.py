import os
from PIL import Image

import numpy as np
import h5py

import main.plot_utils as pu


class InintialConditions:

    def __init__(self, image_path, N, MASS=1, R_CM=[0, 0], V_CM=[0, 0], invert=True):
        """
        Initializes the class with image path, particle number N, default mass, center of mass position and velocity, and an invert flag. Loads image, processes to grayscale, normalizes, optionally inverts, and samples initial positions and colors from the image data.
        """

        self.image_path = image_path
        self.N = int(N)
        self.MASS = MASS
        self.R_CM = np.array(R_CM)
        self.V_CM = np.array(V_CM)

        img_grey = Image.open(image_path).convert("L")  # Reads in greyscale
        color_array = np.array(Image.open(image_path))[::-1, :]

        self.img_array = np.array(img_grey)[::-1, :]
        self.img_array = self.img_array / np.sum(self.img_array)  # Normalise to [0, 1]
        if invert:
            self.img_array = np.max(self.img_array) - self.img_array

        self.cum_sum = np.cumsum(self.img_array.ravel())
        self.cum_sum /= self.cum_sum[-1]

        x, y = self._sample_positions()

        self.colors = color_array[y, x] / 255

        self.POS = np.array([x, y]).T
        self.POS = self.POS / np.max((np.max(x), np.max(y)))  # Normalise to [0, 1]
        self.POS = self.POS - np.mean(self.POS, axis=0)  # Centre around 0

        self.POS = self.POS + self.R_CM

        self.VEL = np.zeros((self.N, 2))
        self.M = self.MASS * np.ones(self.N) / self.N

        self.plot_params_set = False

        x_min, x_max = np.min(self.POS[:, 0]), np.max(self.POS[:, 0])
        y_min, y_max = np.min(self.POS[:, 1]), np.max(self.POS[:, 1])
        area = (x_max - x_min) * (y_max - y_min)
        particle_density = self.N / area
        mean_intparticle_distance = 1 / np.sqrt(particle_density)
        print(f"Mean interparticle distance: {mean_intparticle_distance:.2e}")

    def set_circular_velocity(self, factor=1):
        """
        Calculates circular velocities based on radial distances from the center of mass and applies these to the particle velocities, scaling by a specified factor.
        """
        pos_cm = self.POS - self.R_CM
        radii = np.linalg.norm(self.POS, axis=1)
        comparison_matrix = radii[:, None] >= radii
        count_inside = np.sum(comparison_matrix, axis=1) - 1
        velocity_magnitude = factor * np.sqrt(
            self.MASS * count_inside / (self.N * radii)
        )
        velocity_direction = (
            np.column_stack((-pos_cm[:, 1], pos_cm[:, 0])) / radii[:, None]
        )
        self.VEL = velocity_magnitude[:, None] * velocity_direction

    def generate_ic_file(self, savepath):
        """
        Generates an HDF5 file with initial conditions including particle positions, velocities, and masses. Ensures plot parameters are set and creates necessary directories.
        """
        self.VEL = self.VEL - np.mean(self.VEL, axis=0)
        self.VEL = self.VEL + self.V_CM

        if not self.plot_params_set:
            raise ValueError(
                "Please set plot parameters by calling plot() method before generating initial conditions file"
            )

        savefold = "/".join(savepath.split("/")[:-1])
        if not os.path.exists(savefold):
            os.makedirs(savefold)

        with h5py.File(savepath, "w") as f:
            header_grp = f.create_group("Header")
            header_grp.attrs["Dimensions"] = np.int32(2)
            header_grp.attrs["N"] = np.int32(self.N)

            header_grp.attrs["RCM"] = np.array(self.R_CM, dtype=np.float32)
            header_grp.attrs["VCM"] = np.array(self.V_CM, dtype=np.float32)

            part_type_grp = f.create_group("Bodies")
            part_type_grp.create_dataset(
                "Positions", data=np.array(self.POS, dtype=np.float32)
            )
            part_type_grp.create_dataset(
                "Velocities", data=np.array(self.VEL, dtype=np.float32)
            )
            part_type_grp.create_dataset(
                "Masses", data=np.array(self.M, dtype=np.float32)
            )

            # Some additional attributes for plotting simulation output later
            plot_type_grp = f.create_group("PlotParams")
            plot_type_grp.attrs["MarkerSize"] = np.float32(self.marker_size)
            plot_type_grp.attrs["FaceColor"] = self.facecolor
            plot_type_grp.attrs["AxColor"] = self.ax_color
            plot_type_grp.attrs["LimX"] = np.array(self.lim_x, dtype=np.float32)
            plot_type_grp.attrs["LimY"] = np.array(self.lim_y, dtype=np.float32)
            plot_type_grp.attrs["AxSpines"] = np.int32(self.ax_spines)

            if self.colors is not None:
                plot_type_grp.attrs["HasColor"] = np.int32(1)
                plot_type_grp.create_dataset(
                    "Colors", data=np.array(self.colors, dtype=np.float32)
                )
            else:
                plot_type_grp.attrs["HasColor"] = np.int32(0)

    def set_plot(
        self,
        marker_size=0.1,
        facecolor="#fffff0",
        ax_color="k",
        lim=0.65,
        greyscale=False,
        ax_spines=True,
        show=True,
    ):
        """
        Configures plot parameters such as color, axis limits, and visibility. Optionally displays the plot with customized aesthetics.
        """
        self.ax_spines = np.int32(ax_spines)

        self.plot_params_set = True
        self.facecolor = facecolor
        self.ax_color = ax_color

        if isinstance(lim, (float, int)):
            self.lim_x = [-lim + self.R_CM[0], lim + self.R_CM[0]]
            self.lim_y = [-lim + self.R_CM[1], lim + self.R_CM[1]]

        colors = "k"
        if not greyscale:
            colors = self.colors
            self.colors = colors
        else:
            self.colors = None

        if show:
            Fig = pu.Figure(fig_size=540)
            fs = Fig.fs
            Fig.facecolor = facecolor
            Fig.ax_color = ax_color

            ax = Fig.get_axes()

            self.marker_size = marker_size
            ax.scatter(
                self.POS[:, 0],
                self.POS[:, 1],
                lw=0,
                s=fs * marker_size,
                c=colors,
            )

            ax.set_aspect("equal")

            ax.set_xlim(self.lim_x)
            ax.set_ylim(self.lim_y)

            Fig.fig.subplots_adjust(
                top=1, bottom=0, right=1, left=0, hspace=0, wspace=0
            )

            if not ax_spines:
                ax.set_xticks([])
                ax.set_yticks([])

                for spine in ax.spines.values():
                    spine.set_visible(False)

    def _sample_positions(self):
        """
        Internal method to sample particle positions based on the cumulative distribution of the processed image data.
        """
        random_values = np.random.rand(self.N)
        max_index = len(self.cum_sum) - 1
        indices = np.searchsorted(self.cum_sum, random_values)
        indices = np.clip(indices, 0, max_index)
        y, x = np.unravel_index(indices, self.img_array.shape)
        x, y = np.array(x), np.array(y)
        return x, y


def combine_ics(path_list, path_ics):

    M = len(path_list)

    with h5py.File(path_ics, "w") as combined_f:
        header_grp = combined_f.create_group("Header")
        header_grp.attrs["Dimensions"] = np.int32(2)

        N = 0
        R_CM = np.zeros(2)
        V_CM = np.zeros(2)

        POS = np.empty((0, 2))
        VEL = np.empty((0, 2))
        MASS = np.empty(0)

        marker_size = 0
        facecolor = None
        ax_color = None
        lim_x_mins = []
        lim_x_maxs = []
        lim_y_mins = []
        lim_y_maxs = []
        ax_spines = None
        colors = np.empty((0, 3))

        for i, path in enumerate(path_list):
            with h5py.File(path, "r") as f:
                N_ = f["Header"].attrs["N"]
                N += N_
                R_CM += f["Header"].attrs["RCM"]
                V_CM += f["Header"].attrs["VCM"]

                pos = f["Bodies"]["Positions"][()]
                vel = f["Bodies"]["Velocities"][()]
                mass = f["Bodies"]["Masses"][()]

                POS = np.vstack((POS, pos))
                VEL = np.vstack((VEL, vel))
                MASS = np.hstack((MASS, mass))

                ms = f["PlotParams"].attrs["MarkerSize"]
                marker_size += ms

                if i == 0:
                    facecolor = f["PlotParams"].attrs["FaceColor"]
                    ax_color = f["PlotParams"].attrs["AxColor"]

                lim_x_min = f["PlotParams"].attrs["LimX"][0]
                lim_x_max = f["PlotParams"].attrs["LimX"][1]
                lim_y_min = f["PlotParams"].attrs["LimY"][0]
                lim_y_max = f["PlotParams"].attrs["LimY"][1]

                lim_x_mins.append(lim_x_min)
                lim_x_maxs.append(lim_x_max)
                lim_y_mins.append(lim_y_min)
                lim_y_maxs.append(lim_y_max)

                has_color = f["PlotParams"].attrs["HasColor"]
                if has_color:
                    cs = f["PlotParams"]["Colors"]
                else:
                    cs = np.zeros(N_)

                colors = np.vstack((colors, cs))

        R_CM /= M
        POS = POS - R_CM
        V_CM /= M
        VEL = VEL - V_CM

        header_grp.attrs["N"] = np.int32(N)
        header_grp.attrs["RCM"] = np.array(R_CM, dtype=np.float32)
        header_grp.attrs["VCM"] = np.array(V_CM, dtype=np.float32)

        part_type_grp = combined_f.create_group("Bodies")
        part_type_grp.create_dataset("Positions", data=np.array(POS, dtype=np.float32))
        part_type_grp.create_dataset("Velocities", data=np.array(VEL, dtype=np.float32))
        part_type_grp.create_dataset("Masses", data=np.array(MASS, dtype=np.float32))

        plot_type_grp = combined_f.create_group("PlotParams")
        plot_type_grp.attrs["MarkerSize"] = np.float32(marker_size / M)
        plot_type_grp.attrs["FaceColor"] = facecolor
        plot_type_grp.attrs["AxColor"] = ax_color
        plot_type_grp.attrs["AxSpines"] = np.int32(0)

        lim_x = [np.min(lim_x_mins), np.max(lim_x_maxs)]
        lim_y = [np.min(lim_y_mins), np.max(lim_y_maxs)]
        plot_type_grp.attrs["LimX"] = np.array(lim_x, dtype=np.float32)
        plot_type_grp.attrs["LimY"] = np.array(lim_y, dtype=np.float32)

        plot_type_grp.attrs["HasColor"] = np.int32(1)
        plot_type_grp.create_dataset("Colors", data=np.array(colors, dtype=np.float32))
