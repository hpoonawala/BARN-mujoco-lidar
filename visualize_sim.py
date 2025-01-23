import mujoco
import mujoco.viewer
import numpy as np
import time
import controllers
from math import sin, pi, tanh, exp, atan2
import argparse
import shutil

# Whether to enable gravity compensation.
gravity_compensation: bool = False

# Simulation timestep in seconds.
dt: float = 0.002

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
    
    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=True,
        show_right_ui=False,
    ) as viewer:
        # Reset the simulation.
        # mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Enable site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        # print(model.site()) ## hack that lists the site names, but throws an error

        loop_count=0;
        while viewer.is_running():
            step_start = time.time()
            loop_count+=1
            v,w = controllers.barn2(data.sensordata,weights)

            ## Convert to wheel speed targets
            data.ctrl[0] = v-1.1*w
            data.ctrl[1] = v+1.1*w

            ## Step model and sync time
            mujoco.mj_step(model, data)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process files with sequential numbering")
    parser.add_argument("fnum", type=int, help="Starting integer for file sequence")
    parser.add_argument("--x0", type=float, default=0.0, help="Initial x (default=0.0")
    parser.add_argument("--y0", type=float, default=0.0, help="Initial y (default=0.0")
    parser.add_argument("--t0", type=float, default=0.0, help="Initial theta in deg (default=0.0")
    parser.add_argument("--xml", type=str, default="differentialdrive.xml", help="XML to use for environment")
    args = parser.parse_args()
    print("World number:",args.fnum)
    print("XML File:",args.xml)
    print("Initial:",args.x0,args.y0,args.t0)
    ## Based on input number, load in corresponding world
    shutil.copy("worlds/world_"+str(args.fnum)+".xml","currentxml.xml")
    main(args)
