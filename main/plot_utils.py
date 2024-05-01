import os
import shutil
import time
import subprocess
import PIL

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py


class Figure:

    def __init__(
        self,
        subplot_1=1,
        subplot_2=1,
        fig_size=720,
        ratio=1,
        dpi=300,
        width_ratios=None,
        height_ratios=None,
        hspace=None,
        wspace=None,
    ):
        """
        Initializes a matplotlib figure with configurable subplots, figure size, DPI, aspect ratios, and spacing. Sets up figure and axes aesthetics like colors and grid settings.
        """
        self.fig_size = fig_size
        self.ratio = ratio
        self.dpi = dpi
        self.subplot_1 = subplot_1
        self.subplot_2 = subplot_2
        self.width_ratios = width_ratios
        self.height_ratios = height_ratios
        self.hspace = hspace
        self.wspace = wspace

        fig_width, fig_height = fig_size * ratio / dpi, fig_size / dpi
        fs = np.sqrt(fig_width * fig_height)
        self.fs = fs

        self.fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

        self.ts = 2
        self.sw = 0.2
        self.pad = 0.21
        self.minor_ticks = True
        self.grid = False
        self.ax_color = "w"
        self.facecolor = "k"
        self.text_color = "w"

    def get_axes(self, flat=False):
        """
        Generates and returns the axes of the figure based on the subplot configuration. Optionally returns all axes as a flat list.
        """
        plt.rcParams.update({"text.color": self.text_color})
        self.fig.patch.set_facecolor(self.facecolor)

        subplots = (self.subplot_1, self.subplot_2)
        self.subplots = subplots
        self.gs = mpl.gridspec.GridSpec(
            nrows=subplots[0],
            ncols=subplots[1],
            figure=self.fig,
            width_ratios=self.width_ratios or [1] * subplots[1],
            height_ratios=self.height_ratios or [1] * subplots[0],
            hspace=self.hspace,
            wspace=self.wspace,
        )

        self.axes = []
        for i in range(self.subplots[0]):
            row_axes = []
            for j in range(self.subplots[1]):
                ax = self.fig.add_subplot(self.gs[i, j])
                row_axes.append(ax)
                self.customize_axes(ax)

                if self.hspace == 0 and i != self.subplots[0] - 1:
                    ax.set_xticklabels([])

            self.axes.append(row_axes)

        if self.subplot_1 == 1 and self.subplot_2 == 1:
            return self.axes[0][0]

        self.axes_flat = [ax for row in self.axes for ax in row]

        if flat:
            return self.axes_flat
        else:
            return self.axes

    def customize_axes(
        self,
        ax,
    ):
        """
        Applies custom settings to axes, including tick parameters, colors, grid visibility, and minor tick settings.
        """
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=self.ts * self.fs,
            size=self.fs * self.sw * 5,
            width=self.fs * self.sw * 0.9,
            pad=self.pad * self.fs,
            top=True,
            right=True,
            direction="inout",
            color=self.ax_color,
            labelcolor=self.ax_color,
        )

        if self.minor_ticks == True:
            ax.minorticks_on()

            ax.tick_params(
                axis="both",
                which="minor",
                direction="inout",
                top=True,
                right=True,
                size=self.fs * self.sw * 2.5,
                width=self.fs * self.sw * 0.8,
                color=self.ax_color,
            )

        ax.set_facecolor(self.facecolor)

        for spine in ax.spines.values():
            spine.set_linewidth(self.fs * self.sw)
            spine.set_color(self.ax_color)

        if self.grid:

            ax.grid(
                which="major",
                linewidth=self.fs * self.sw * 0.5,
                color=self.ax_color,
                alpha=0.25,
            )

    def save(self, path, bbox_inches="tight", pad_inches=None):
        """
        Saves the figure to the specified path with options for DPI and padding adjustments.
        """
        self.fig.savefig(
            path, dpi=self.dpi, bbox_inches=bbox_inches, pad_inches=pad_inches
        )

        self.path = path


def make_animation(
    path_output,
    ratio=1,
    fig_size=1400,
    lim=None,
    marker_color=None,
    marker_size=None,
    facecolor=None,
    ax_color=None,
    ax_spines=None,
    fps=30,
    delete_frames=False,
    reverse=False,
    format="mp4",
):

    if isinstance(lim, (float, int)):
        lim_x = [-lim, lim]
        lim_y = [-lim, lim]

    # get folder of the data_path
    savefold = path_output.split("/")[:-1]
    savefold = "/".join(savefold) + "/frames/"

    with h5py.File(path_output, "r") as file:

        N = file["Header"].attrs["N"]
        M = file["Header"].attrs["NSnapshots"]
        positions = np.zeros((M, N, file["Header"].attrs["Dimensions"]))

        if marker_color is None:
            has_color = file["PlotParams"].attrs["HasColor"]
            if has_color:
                marker_color = file["PlotParams"]["Colors"][()]
            else:
                marker_color = "k"

        if marker_size is None:
            marker_size = file["PlotParams"].attrs["MarkerSize"]
        if facecolor is None:
            facecolor = file["PlotParams"].attrs["FaceColor"]
        if ax_color == None:
            ax_color = file["PlotParams"].attrs["AxColor"]

        if lim is None:
            lim_x = file["PlotParams"].attrs["LimX"]
            lim_y = file["PlotParams"].attrs["LimY"]
        if ax_spines is None:
            ax_spines = file["PlotParams"].attrs["AxSpines"]

        for i in range(M):
            positions[i] = file[f"{i:04d}"]["Positions"]

    Fig = Figure(ratio=ratio, fig_size=fig_size)
    Fig.facecolor = facecolor
    Fig.ax_color = ax_color
    ax = Fig.get_axes()
    fs = Fig.fs

    ax.axis("equal")

    ax.set_xlim(lim_x)
    ax.set_ylim(lim_y)

    Fig.fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    if not os.path.exists(savefold):
        os.makedirs(savefold)
    else:
        shutil.rmtree(savefold)
        os.makedirs(savefold)

    time_start = time.time()
    for ii, i in enumerate(range(0, M)):

        scatter = ax.scatter(
            positions[i, :, 0],
            positions[i, :, 1],
            c=marker_color,
            s=fs * marker_size,
            lw=0.0 * fs,
            alpha=1,
        )

        fig_name = f"render_{ii:04d}.jpg"
        save_path = savefold + fig_name

        if ax_spines:

            Fig.save(save_path)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(False)

            Fig.fig.patch.set_facecolor("grey")

            Fig.save(save_path, bbox_inches="tight", pad_inches=0)

        plt.close()

        scatter.remove()
    print(f"Save images time: {time.time() - time_start:.2f} s")

    time_start = time.time()

    if format == "gif":
        png_to_gif(savefold, fps=fps, reverse=reverse)
    else:
        png_to_mp4(savefold, fps=fps, reverse=reverse)

    print(f"Video creation time: {time.time() - time_start:.2f} s")

    if delete_frames:
        shutil.rmtree(savefold)

    return


def png_to_mp4(
    fold,
    title="video",
    fps=36,
    digit_format="04d",
    res=None,
    resize_factor=1,
    custom_bitrate=None,
    extension=".jpg",
    reverse=False,  # Adding reverse parameter with default value False
):

    # Get a list of all image files in the directory with the specified extension
    files = [f for f in os.listdir(fold) if f.endswith(extension)]
    files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    if not files:
        raise ValueError("No image files found in the specified folder.")

    im = PIL.Image.open(os.path.join(fold, files[0]))
    resx, resy = im.size

    if res is not None:
        resx, resy = res
    else:
        resx = int(resize_factor * resx)
        resy = int(resize_factor * resy)
        resx += resx % 2  # Ensuring even dimensions
        resy += resy % 2

    basename = os.path.splitext(files[0])[0].split("_")[0]

    ffmpeg_path = "ffmpeg"
    abs_path = os.path.abspath(fold)
    parent_folder = os.path.dirname(abs_path) + os.sep
    output_file = os.path.join(parent_folder, f"{title}.mp4")

    crf = 5  # Lower CRF for higher quality, higher for lower quality
    bitrate = custom_bitrate if custom_bitrate else "5000k"
    preset = "slow"
    tune = "film"

    # Construct the ffmpeg command
    command = f'{ffmpeg_path} -y -r {fps} -i {os.path.join(fold, f"{basename}_%{digit_format}{extension}")} -c:v libx264 -profile:v high -crf {crf} -preset {preset} -tune {tune} -b:v {bitrate} -pix_fmt yuv420p -vf "scale={resx}:{resy}'
    if reverse:
        command += ",reverse"  # Appends the reverse filter if reverse is True
    command += f'" {output_file}'

    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error during video conversion:", e)


def png_to_gif(
    fold,
    title="video",
    outfold=None,
    fps=24,
    digit_format="04d",
    quality=500,
    max_colors=256,
    extension=".jpg",
    reverse=False,  # Adding reverse parameter with default value False
):

    files = [f for f in os.listdir(fold) if f.endswith(extension)]
    files.sort()

    name = os.path.splitext(files[0])[0]
    basename = name.split("_")[0]

    ffmpeg_path = "ffmpeg"
    framerate = fps

    if outfold is None:
        abs_path = os.path.abspath(fold)
        parent_folder = os.path.dirname(abs_path) + "\\"
    else:
        parent_folder = outfold
        if not os.path.exists(parent_folder):
            os.makedirs(parent_folder)

    output_file = parent_folder + "{}.gif".format(title)

    # Create a palette with limited colors for better file size
    palette_file = parent_folder + "palette.png"
    palette_command = f'{ffmpeg_path} -i {fold}{basename}_%{digit_format}{extension} -vf "fps={framerate},scale={quality}:-1:flags=lanczos,palettegen=max_colors={max_colors}" -y {palette_file}'
    subprocess.run(palette_command, shell=True)

    # set paletteuse
    paletteuse = "paletteuse=dither=bayer:bayer_scale=5"

    # Construct video filter with conditional reverse
    video_filters = (
        f"fps={framerate},scale={quality}:-1:flags=lanczos [x]; [x][1:v] {paletteuse}"
    )
    if reverse:
        video_filters = f"fps={framerate},scale={quality}:-1:flags=lanczos,reverse [x]; [x][1:v] {paletteuse}"

    # Use the optimized palette to create the GIF
    gif_command = f'{ffmpeg_path} -r {framerate} -i {fold}{basename}_%04d{extension} -i {palette_file} -lavfi "{video_filters}" -y {output_file}'
    subprocess.run(gif_command, shell=True)

    # delete palette
    os.remove(palette_file)
