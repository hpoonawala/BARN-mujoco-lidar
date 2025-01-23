import numpy as np
from math import cos, sin, pi, tanh, exp, atan2
import argparse
import shutil
import scan_match
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

def invert_delta(dx,dy,dt):
    """
    ## (R,T) -> (R',-R'T). In SE(2), R -> R' is equiv to flipping sign of single Euler angle
    """
    return -dx*cos(dt)-dy*sin(dt),dx*sin(dt)-dy*cos(dt),-dt


def odom(args):
    """
    loads the stored scans and their poses (gen in MuJoCo) and solves for relative positions
    Saves the odometry estimates for fast loading 
    """
    fname = args.prefix+str(args.fnum)+"_scans"+args.suffix
    data = np.load(fname)
    scanlist = [data[k] for k in data]

    fname = args.prefix+str(args.fnum)+"_poses"+args.suffix
    data = np.load(fname)
    poselist = [data[k] for k in data] # true poses from sim
    ## use the first pose as the initial pose, ignoring passed arguments
    args.x0=poselist[0][0]
    args.y0=poselist[0][1]
    args.t0=poselist[0][2]
    ## Now, we can call match_consecutive_scans 
    manual_rel_poses = match_consecutive_scans(scanlist,poselist,args)
    odom_poses = integrate_path(manual_rel_poses,args)
    return scanlist, poselist, odom_poses
    # print(poselist)
    # print(odom_poses)

def match_consecutive_scans(sensors,poses,args): 
    """
    # Gets a sequence of relative poses using the Normal Distribution Transform for scan matching
    """ 
    manual_rel_poses = np.zeros((len(sensors),3)) ## pose differences using scan matching
    for i in range(1,len(sensors)): ## pair-wise comparison here
        print(f"Match {i} and {i-1}")
        if len(sensors[i-1])>0 and len(sensors[i])>0: ## non-trivial returns
            dx = (poses[i][0]-poses[i-1][0])*(cos(poses[i-1][2]))+(poses[i][1]-poses[i-1][1])*(sin(poses[i-1][2]))
            dy = -(poses[i][0]-poses[i-1][0])*(sin(poses[i-1][2]))+(poses[i][1]-poses[i-1][1])*(cos(poses[i-1][2]))
            dt = poses[i][2]-poses[i-1][2]
            print("true rel pose",dx,dy,dt)
            match_result, match_cov=scan_match.ndt_scan_match_hp(sensors[i],sensors[i-1],2.0) ## find parameters needed to push prev scan to current scan
            manual_rel_poses[i][0] = match_result["translation"][0] 
            manual_rel_poses[i][1] = match_result["translation"][1]
            manual_rel_poses[i][2] = match_result["rotation"]
            print("scanmatch result hp:",manual_rel_poses[i])
            ## Motion of scan is body frame is opposite of motion of body in world frame. Invert rel pose:
            dx, dy, dt = invert_delta(manual_rel_poses[i][0],manual_rel_poses[i][1],manual_rel_poses[i][2])
            manual_rel_poses[i][0] = dx
            manual_rel_poses[i][1] = dy
            manual_rel_poses[i][2] = dt
            print("est rel pose:",manual_rel_poses[i])


            ## to visually debug scan matching:
            # transformed_scan=scan_match.transform_scan(sensors[i-1],match_result["translation"][0],match_result["translation"][1],match_result["rotation"])
            # fig = plt.figure();
            # ax = plt.subplot(111)
            # ndt_grid = scan_match.compute_ndt_grid(sensors[i], 2.0)
            # for g in ndt_grid:
            #     scan_match.confidence_ellipse(ndt_grid[g][0],ndt_grid[g][1] , ax,3,alpha=0.5, facecolor='pink')
            #     ax.scatter(ndt_grid[g][0][0],ndt_grid[g][0][1],color='y')
            # x1, y1 = zip(*sensors[i-1])
            # x2, y2 = zip(*sensors[i])
            # x3, y3 = zip(*transformed_scan)
            # ax.scatter(x1,y1,color='r',label=f"scan at {i-1} in body frame at {i-1}")
            # ax.scatter(x2,y2,color='b',label=f"scan at {i} in body frame at {i}")
            # ax.scatter(x3,y3,color='c',label=f"scan at {i-1} in body frame at {i}")
            # plt.legend()
            # plt.show(block=False)
            # plt.pause(0.1)
            # plt.close()

    ## Save the relative odometry, which is the bulk of the work
    np.savez(args.prefix+str(args.fnum)+"_relodom"+args.suffix, *manual_rel_poses)
    return manual_rel_poses
                

def load_rel_odom(args):
    """
    ## loads the stored scans and their true poses and odometry-estimated poses, possible calculated from `odom`
    """
    fname = args.prefix+str(args.fnum)+"_scans"+args.suffix
    data = np.load(fname)
    scanlist = [data[k] for k in data]

    fname = args.prefix+str(args.fnum)+"_poses"+args.suffix
    data = np.load(fname)
    poselist = [data[k] for k in data] # true poses from sim

    ## use the first pose as the initial pose, ignoring passed arguments
    args.x0=poselist[0][0]
    args.y0=poselist[0][1]
    args.t0=poselist[0][2]

    fname = args.prefix+str(args.fnum)+"_relodom"+args.suffix
    data = np.load(fname)
    rel_poses = [data[k] for k in data]
    return scanlist, poselist, rel_poses

def integrate_path(manual_rel_poses,args): 
    path_integ = np.array([args.x0,args.y0,args.t0]) ## path_integ is just the global pose over time in the 'world' frame 
    odom_poses = np.zeros((len(manual_rel_poses)+1,3))
    odom_poses[0] = np.array([args.x0,args.y0,args.t0])
    for count in range(0,len(manual_rel_poses)-1):
        path_integ[0]+=manual_rel_poses[count+1][0]*cos(path_integ[2]) - manual_rel_poses[count+1][1]*sin(path_integ[2]) 
        path_integ[1]+=manual_rel_poses[count+1][0]*sin(path_integ[2]) + manual_rel_poses[count+1][1]*cos(path_integ[2]) 
        path_integ[2]+=manual_rel_poses[count+1][2]
        odom_poses[count+1][0] = path_integ[0]
        odom_poses[count+1][1] = path_integ[1]
        odom_poses[count+1][2] = path_integ[2]
    np.savez(args.prefix+str(args.fnum)+"_odom"+args.suffix, *odom_poses)
    return odom_poses



def main(args) -> None:
    ## If odometry needs to be (re)done:
    scanlist, poselist, odom_poses = odom(args) 
    ## If odometry has been done before, can view results of some new integration using:
    # scanlist, poselist, rel_poses = load_rel_odom(args) 
    # odom_poses = integrate_path(rel_poses,args)
    fig = plt.figure();
    ax = plt.subplot(111)
    for p,o in zip(poselist,odom_poses):
        ax.scatter(p[0],p[1],color='r')
        ax.scatter(o[0],o[1],color='b')

    ax.scatter(poselist[0][0],poselist[0][1],color='r',label="ground truth")
    ax.scatter(odom_poses[0][0],odom_poses[0][1],color='b',label="odometry")
    plt.legend()
    fig.savefig(args.prefix+str(args.fnum)+'_odomplot.png')
    plt.show()

    # scanlist, poselist, rel_poses = load_rel_odom(args) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process multiple files with sequential numbering")
    parser.add_argument("fnum", type=int, help="Starting integer for file sequence")
    parser.add_argument("--x0", type=float, default=0.0, help="Initial x (default=0.0")
    parser.add_argument("--y0", type=float, default=0.0, help="Initial y (default=0.0")
    parser.add_argument("--t0", type=float, default=0.0, help="Initial theta in deg (default=0.0")
    parser.add_argument("--prefix", type=str, default="simdata/sim_", help="Prefix for the files (default: 'simdata/sim_')")
    parser.add_argument("--suffix", type=str, default=".npz", help="Suffix for the files (default: '.npz')")
    args = parser.parse_args()
    # print(args.fnum,args.prefix,args.suffix)
    main(args)
