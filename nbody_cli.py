import argparse
import os
import json
from datetime import datetime
from dataclasses import dataclass, asdict

import main.plot_utils as pu
from main.make_ics import InintialConditions
from main.simulate import NBodySimulation


@dataclass
class SimulationConfig:
    image_path: str = "sample_images/einstein.jpg"
    N: float = 5e3
    MASS: float = 2
    R_CM: list = (0, 0)
    V_CM: list = (0, 0)
    invert: bool = True
    factor: float = 0.5
    marker_size: float = 0.75
    lim: float = 0.55
    facecolor: str = "#ebe9d8"
    ax_color: str = "k"
    greyscale: bool = False
    ax_spines: bool = False
    show: bool = True
    T: float = 3
    dt: float = 0.01
    num_snapshots: int = 300
    fps: int = 30
    delete_frames: bool = False
    reverse: bool = True
    format: str = ".mp4"


def main(config: SimulationConfig):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    simulation_folder = f"results/{timestamp}/"
    os.makedirs(simulation_folder, exist_ok=True)

    # Write configuration to JSON file
    with open(os.path.join(simulation_folder, 'config.json'), 'w') as config_file:
        json.dump(asdict(config), config_file, indent=4, sort_keys=True)

    Ics = InintialConditions(config.image_path, config.N, MASS=config.MASS, R_CM=config.R_CM, V_CM=config.V_CM, invert=config.invert)
    Ics.set_circular_velocity(factor=config.factor)
    Ics.set_plot(marker_size=config.marker_size, lim=config.lim, facecolor=config.facecolor, ax_color=config.ax_color, greyscale=config.greyscale,
                 ax_spines=config.ax_spines, show=config.show)

    path_ics = os.path.join(simulation_folder, "initial_conditions.hdf5")
    Ics.generate_ic_file(path_ics)

    path_output = os.path.join(simulation_folder, 'output.hdf5')
    Sim = NBodySimulation(path_ics, path_output)
    Sim.set_time(config.T, config.dt, snapshots=config.num_snapshots)
    Sim.estimate_runtime(num_steps=3)
    Sim.run_simulation(0.05)

    pu.make_animation(
        path_output,
        fig_size=720,
        ratio=1,
        fps=config.fps,
        delete_frames=config.delete_frames,
        reverse=config.reverse,
        format=config.format,
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

    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    config = SimulationConfig(**args_dict)

    main(config)
