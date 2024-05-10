# nbody_whatever

![Example](example.gif)

A toy N-body simulator. Use an image for your initial conditions by converting it into N bodies. The simulator uses a brute force approach (at O(N^2) force calculations per time step) and it is parallelized with numba. Snapshots from the simulation are saved at predefined times as an `.hdf5` file. Initial conditions are also stored as `.hdf5`. 

## Structure

- `main/`: Main module.
    - `main/make_ics.py`: Contains the `InitialConditions` class for reading the image and converting it to N bodies. 
    - `main/simulate.py`: Contains the `NBodySimulation` class for performing the simulation.
    - `main/plot_utils.py`: Utility functions for plotting and making animations of the simulated output. 


## Usage

Check the notebook `example_usage.ipynb`.

## Requirements

Check requirements.txt, but mainly: `numpy`, `matplotlib`, `PIL`, `h5py` (for storing data) and `numba` (for parallelization) 

## CLI Usage  

Command:  
`python nbody_cli.py --image_path "path/to/your/image.png" --N 100`

When run via cli, the resulting video will stored inside 'results/_timestamp_' folder, along with a json which capture the 
configuration/parameters that were applied at the time of the simulation.