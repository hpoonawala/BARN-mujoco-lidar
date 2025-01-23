
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv


def printsim(args):

    fname = args.prefix+str(args.fnum)+args.suffix
    data = np.load(fname)
    qpos_trajectory = data["qpos"] # Unused
    qvel_trajectory = data["qvel"] # each row is position of marker frame followed by quaternion (len=7)
    ctrl_trajectory = data["ctrl"] # wheel velocities
    mind_trajectory = data["mind"] # minimum return

    num_steps = qpos_trajectory.shape[0]
    time = np.arange(num_steps)  # Time steps

    # print("final values: pos",qvel_trajectory[49999][0:3], " v: ", ctrl_trajectory[49999][0], " w: ", ctrl_trajectory[49999][1])
    print(args.fnum, ": minimum distance: ",np.min(mind_trajectory[1:]), np.max([ctrl_trajectory[num_steps-1][0],-ctrl_trajectory[num_steps-1][0]]) > 1e-1, np.max([qvel_trajectory[num_steps-1][1],-qvel_trajectory[num_steps-1][1]]) > 15) 
    # return np.min(mind_trajectory[1:]), np.max([ctrl_trajectory[49999][0],-ctrl_trajectory[49999][0]]) > 1e-1 && np.max([qvel_trajectory[49999][1],-qvel_trajectory[49999][1]]) > 15
    file = open(args.prefix+"output.txt","a")
    line=str(args.fnum) + ": minimum distance: "+str(np.min(mind_trajectory[1:]))+ " "+ str(np.max([qvel_trajectory[num_steps-1][1],-qvel_trajectory[num_steps-1][1]]) > 15)

    file.write(line + "\n")
    file.close()


def plotsim(args):
    fname = args.prefix+str(args.fnum)+args.suffix
    data = np.load(fname)
    qpos_trajectory = data["qpos"] # Unused
    qvel_trajectory = data["qvel"] # each row is position of marker frame followed by quaternion (len=7)
    ctrl_trajectory = data["ctrl"] # wheel velocities
    mind_trajectory = data["mind"] # minimum return

    num_steps = qpos_trajectory.shape[0]
    time = np.arange(num_steps)  # Time steps


    # Plot phase plane
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    ax.plot(qvel_trajectory[:,0],qvel_trajectory[:,1])
    csv_filename = "worlds/world_"+str(args.fnum)+".csv"
    with open(csv_filename, newline='\n') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            circle = plt.Circle((2*float(row[0])+3, 2*float(row[1])-3), 0.2, color='r')
            ax.add_patch(circle)

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Phase Plane")
    plt.savefig(args.prefix+str(args.fnum)+"_plot.png")
    # plt.show()


    # Plot control
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    ax1.plot(time, ctrl_trajectory[:, 0])
    ax2.plot(time, ctrl_trajectory[:, 1])
    # for joint_idx in range(2):
    #     plt.plot(time, ctrl_trajectory[:, joint_idx], label=f'Control {joint_idx + 1}')
    plt.xlabel("Time step")
    plt.savefig(args.prefix+str(args.fnum)+"_ctrl.png")
    # plt.show()



    # Plot min_d
    plt.figure(figsize=(12, 6))
    plt.plot(time[1:], mind_trajectory[1:], label=f'Min distance')
    plt.xlabel("Time step")
    plt.ylabel("min dist")
    plt.legend()
    plt.savefig(args.prefix+str(args.fnum)+"_mind.png")
    # plt.show()

    # print(mind_trajectory)
    # # Plot control inputs
    # plt.figure(figsize=(12, 6))
    # for joint_idx in range(ctrl_trajectory.shape[1]):
    #     plt.plot(time, ctrl_trajectory[:, joint_idx], label=f'Joint {joint_idx + 1}')
    # plt.xlabel("Time step")
    # plt.ylabel("Control Input (torque)")
    # plt.title("Control Inputs over Time")
    # plt.legend()
    # plt.show()

def main(args) -> None:
    plotsim(args)
    printsim(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process multiple files with sequential numbering")
    parser.add_argument("fnum", type=int, help="Starting integer for file sequence")
    parser.add_argument("--prefix", type=str, default="simdata/sim_", help="Prefix for the files (default: 'simdata/sim_')")
    parser.add_argument("--suffix", type=str, default=".npz", help="Suffix for the files (default: '.npz')")
    args = parser.parse_args()
    # print(args.fnum,args.prefix,args.suffix)
    main(args)
