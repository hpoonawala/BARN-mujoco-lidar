import mujoco
import mujoco.viewer
import numpy as np
import time
from math import cos, sin, pi, tanh, exp, atan2
import argparse
import shutil
import controllers
import matplotlib.pyplot as plt


def qpos2pose(qpos):
    theta = 2*atan2(qpos[6],qpos[3])
    return np.array([qpos[0],qpos[1],theta])

def range2scan(rangescan):
    scan = [] ## initialize a list
    pose = qpos2pose(rangescan[360:367])
    for i in range(360):
        if rangescan[i] < 20.00:
            scan.append([ -rangescan[i]*cos((i-180)*pi/180),rangescan[i]*sin((i-180)*pi/180)]) ## append non-trivial points
            ## The minus sign stems from an angle convention

    ## debug view scan briefly as it is loaded
    # if len(scan)>0:
    #     fig = plt.figure();
    #     ax = plt.subplot(111)
    #     ax.scatter(0,0,color='b')
    #     x_coords, y_coords = zip(*scan)
    #     ax.scatter(x_coords,y_coords,color='r')
    #     plt.show(block=False)
    #     plt.pause(0.1)
    #     plt.close()
    return np.array(scan), pose

# Whether to enable gravity compensation.
gravity_compensation: bool = False

# Simulation timestep in seconds.
dt: float = 0.002

# Lists to store measurements and ground truth pose
scanlist = []
poselist=[]

def main(args) -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("differentialdrive.xml")
    data = mujoco.MjData(model)

    # Enable gravity compensation. Set to 0.0 to disable.
    model.body_gravcomp[:] = float(gravity_compensation)
    model.opt.timestep = dt

    # Set the initial condition
    data.qpos[0] = args.x0
    data.qpos[1] = args.y0
    mujoco.mju_euler2Quat(data.qpos[3:7],[0,0,args.t0*pi/180],"XYZ")

    # Set the weights
    weights=np.zeros(360)
    for i in range(360):
        weights[i] = -sin((i-180)*pi/180)

    # Get the dof and actuator ids for the joints we wish to control. These are copied
    # from the XML file. Feel free to comment out some joints to see the effect on
    # the controller.
    joint_names = [
        "left-wheel",
        "right-wheel",
    ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    # actuator_ids = np.array([model.actuator(name).id for name in joint_names])
    
    qpos_trajectory = []
    qvel_trajectory = []
    ctrl_trajectory = []
    min_dist_trajectory = []

    
    for loop_count in range(args.nsteps):

        if loop_count == 0: ## Overcoming initial all-zeros scan
            v=0.0 # for storage later
            w=0.0
            data.ctrl[0] =0.0
            data.ctrl[1] =0.0

        ## Convert to wheel speed targets
        if (loop_count % 40) == 1: # Update control only at 12.5 Hz, assuming dt = 0.002 seconds
            v,w = controllers.barn2(data.sensordata,weights)
            data.ctrl[0] = v-1.0*w
            data.ctrl[1] = v+1.0*w

        # Store joint positions, velocities, and control inputs at each time step
        qpos_trajectory.append(data.qpos.copy())
        qvel_trajectory.append(data.sensordata[360:367].copy())
        ctrl_trajectory.append(np.array([v,w]))
        min_dist_trajectory.append(np.min(data.sensordata[0:360]))

            
        mujoco.mj_step(model, data)
        if (loop_count % 40) == 1: # Update control only at 12.5 Hz, assuming dt = 0.002 seconds
            newscan,newpose = range2scan(data.sensordata)
            scanlist.append(newscan)
            poselist.append(newpose)


    ## Save the data in files to be used by odometry and mapping code
    np.savez(args.prefix+str(args.fnum)+args.suffix, qpos=qpos_trajectory, qvel=qvel_trajectory, ctrl=ctrl_trajectory,mind=min_dist_trajectory)
    np.savez(args.prefix+str(args.fnum)+"_scans"+args.suffix, *scanlist)
    np.savez(args.prefix+str(args.fnum)+"_poses"+args.suffix, *poselist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process multiple files with sequential numbering")
    parser.add_argument("fnum", type=int, help="Starting integer for file sequence")
    parser.add_argument("--x0", type=float, default=0.0, help="Initial x (default=0.0)")
    parser.add_argument("--y0", type=float, default=0.0, help="Initial y (default=0.0)")
    parser.add_argument("--t0", type=float, default=0.0, help="Initial theta in deg (default=0.0)")
    parser.add_argument("--nsteps", type=int, default=5000,help="Number of sim steps")
    parser.add_argument("--xml", type=str, default="differentialdrive.xml", help="XML to use for environment")
    parser.add_argument("--prefix", type=str, default="simdata/sim_", help="Prefix for the files (default: 'simdata/sim_')")
    parser.add_argument("--suffix", type=str, default=".npz", help="Suffix for the files (default: '.npz')")
    args = parser.parse_args()
    print("World number:",args.fnum)
    print("XML File:",args.xml, "Steps: ",args.nsteps)
    print("Initial:",args.x0,args.y0,args.t0)
    print("Sim file:",args.prefix+str(args.fnum)+args.suffix)
    ## Based on input number, load in corresponding world
    shutil.copy("worlds/world_"+str(args.fnum)+".xml","currentxml.xml")
    main(args)
