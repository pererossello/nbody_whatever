import argparse
import os
from datetime import datetime

import main.plot_utils as pu
from main.make_ics import InintialConditions
from main.simulate import NBodySimulation


def main(image_path, N, MASS, R_CM, V_CM, invert, factor, marker_size, lim, facecolor, ax_color, greyscale, ax_spines, show, T, dt, num_snapshots,
         fps, delete_frames, reverse, format):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    simulation_folder = f"results/{timestamp}/"
    os.makedirs(simulation_folder, exist_ok=True)
    Ics = InintialConditions(image_path, N, MASS=MASS, R_CM=R_CM, V_CM=V_CM, invert=invert)
    Ics.set_circular_velocity(factor=factor)
    Ics.set_plot(marker_size=marker_size, lim=lim, facecolor=facecolor, ax_color=ax_color, greyscale=greyscale, ax_spines=ax_spines, show=show)

    simulation_folder = "results/"
    os.makedirs(simulation_folder, exist_ok=True)
    path_ics = os.path.join(simulation_folder, "initial_conditions.hdf5")
    Ics.generate_ic_file(path_ics)

    path_output = os.path.join(simulation_folder, 'output.hdf5')
    Sim = NBodySimulation(path_ics, path_output)
    Sim.set_time(T, dt, snapshots=num_snapshots)
    Sim.estimate_runtime(num_steps=3)
    Sim.run_simulation(0.05)

    pu.make_animation(
        path_output,
        fig_size=720,
        ratio=1,
        fps=fps,
        delete_frames=delete_frames,
        reverse=reverse,
        format=format,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an N-body simulation based on image data.")
    parser.add_argument("--image_path", type=str, default="sample_images/logo.png", help="Path to the image file.")
    parser.add_argument("--N", type=float, default=5e3, help="Number of bodies.")
    parser.add_argument("--MASS", type=float, default=2, help="Mass of each body.")
    parser.add_argument("--R_CM", type=float, nargs=2, default=[0, 0], help="Center of mass position vector.")
    parser.add_argument("--V_CM", type=float, nargs=2, default=[0, 0], help="Center of mass velocity vector.")
    parser.add_argument("--invert", type=bool, default=True, help="Invert image brightness values.")
    parser.add_argument("--factor", type=float, default=0.5, help="Factor for setting circular velocity.")
    parser.add_argument("--marker_size", type=float, default=0.75, help="Marker size for plots.")
    parser.add_argument("--lim", type=float, default=0.55, help="Plot limits.")
    parser.add_argument("--facecolor", type=str, default="#ebe9d8", help="Background color of the plot.")
    parser.add_argument("--ax_color", type=str, default="k", help="Color of the axes.")
    parser.add_argument("--greyscale", type=bool, default=False, help="Use greyscale for the plot.")
    parser.add_argument("--ax_spines", type=bool, default=False, help="Show axes spines.")
    parser.add_argument("--show", type=bool, default=True, help="Show the plot.")
    parser.add_argument("--T", type=float, default=3, help="Total simulation time.")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step for the simulation.")
    parser.add_argument("--num_snapshots", type=int, default=300, help="Number of snapshots to save.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the animation.")
    parser.add_argument("--delete_frames", type=bool, default=True, help="Delete frames after creating video.")
    parser.add_argument("--reverse", type=bool, default=True, help="Reverse the time in the video.")
    parser.add_argument("--format", type=str, default=".gif", help="Format of the output video (e.g., .mp4, .gif).")
    args = parser.parse_args()

    main(**vars(args))
