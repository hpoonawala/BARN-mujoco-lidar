# LiDAR-based navigation in `MuJoCo` 

This repo contains code to simulate -- **using `MuJoCo`** instead of `ROS` or `Gazebo` -- a simple differential drive robot navigating the [BARN challenge](https://cs.gmu.edu/~xiao/Research/BARN_Challenge/BARN_Challenge25.html) worlds [(download zip file)](https://cs.gmu.edu/~xiao/Research/BARN/BARN_dataset.zip) using a LiDAR sensor for feedback. 

## Requirements

A `requirements.txt` file has been included. The python code uses packages `mujoco`, `numpy`, `scipy`, and `matplotlib`. 


## Simulating a world
Run
```
python3 sim_world.py <world_number>
```

Optional arguments:

- `x0`: initial $x$ position in meters
- `y0`: initial $y$ position in meters
- `t0`: initial $\theta$ angle in degrees
- `nsteps`: number of steps to run sim
- `xml`: which environment to simulate
- `prefix`: prefix of sim file name
- `suffix`: suffix of sim file name

Run this command for defaults:
```
python3 sim_world.py --help
```

Due to the size of the arena, a common choice is:
```
python3 sim_world.py <world_number> --y0=7.0
```

## XML File
The simulation environment of the differential drive robot with LiDAR is defined in `differentialdrive.xml`. 


### Arena
The chosen pre-stored `worlds/world_<world_number>.xml` is copied onto `currentxml.xml` by the `sim_world.py` code. The `differentialdrive.xml` file loads in `currentxml.xml`, bypassing the need to modify `differentialdrive.xml`. 

### Control
Currently, the controller used to map LiDAR readings into body velocities and then wheel speeds are hard-coded, using a controller defined in `controllers.py`. This controller uses a custom sensor-based controller together with a proportional controller acting on the heading error from the $y$-axis direction. You can modify the code to test your own controller, either directly in `sim_world.py` or by calling a new controller placed in `controllers.py`. 

### Sensor readings
The `data.sensordata` array currently has length $367$. The first $360$ entries are the readings from the range sensors, starting from $-180^{\circ}$ relative to forward to $+179^{\circ}$. The next seven floating point numbers are the pose of the `marker` site in the world frame. 

To access LiDAR, `differentialdrive.xml` loads `lidar_frame.xml` and `lidar.xml`. These files currently define a $360$ degree field-of-view LiDAR with resolution of $1$ degree. 

The LiDAR is built by rotating a `rangefinder` sensor about an axis in equal increments using the `<replicate>` tag. The LiDAR can be moved using options for the `<body>` tag in `lidar_frame.xml`. The LiDAR frame currently needs to be rotate by $90$ degrees relative to the world frame to get a horizontal set of rays. The range can be changed using the `cut-off` option in `lidar.xml`. The field-of-view and resolution can be modified by changing the number of replications and the angle increment appropriately. 

## Plots of a simulation
The `sim_world.py` file generates three files with names containing the `<world_number>`. These can be processed to generate images using
```
python3 generate_plot.py <world_number>
```
You may need to give the same options that were given to `sim_world.py` if not using the default ones. 


## Visualizing a simulation
A separate script is used to visualize the simulation. Run:

```
mjpython visualize_sim.py <world_number>
```

Unlike `sim_world.py`, the control frequency is the same as the simulation rate. Also, no simulation data are stored. 

## Odometry
The sequence of scans stored after using `sim_world.py` can be used to estimate the path of the robot through scan matching of consecutive scans. The odometry script is not currently not optimized. Also, no map is generated.  To compute the path, run:

```
python3 odometry.py <world_number>
```
You may need to give the same options that were given to `sim_world.py` if not using the default ones. 

## Helper scripts
The short scripts 

- `run_sim_all_worlds.py`
- `generate_all.py`

enable evaluation of a controller (currently hard-coded) in all worlds, or some desired subset (currently hard-coded). 
