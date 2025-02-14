import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
from math import cos, sin, pi, tanh, exp, atan2
import scan_match 
import copy ## just to compare

def invert_transform(pose):
    ## (R,T) -> (R',-R'T). In SE(2), R -> R' is equiv to flipping sign of single Euler angle
    dx = pose[0]
    dy = pose[1]
    dt = pose[2]
    return -dx*cos(dt)-dy*sin(dt),dx*sin(dt)-dy*cos(dt),-dt

def relative_transform(from_pose,to_pose):
    dx = (to_pose[0]-from_pose[0])*(cos(from_pose[2]))+(to_pose[1]-from_pose[1])*(sin(from_pose[2]))
    dy = -(to_pose[0]-from_pose[0])*(sin(from_pose[2]))+(to_pose[1]-from_pose[1])*(cos(from_pose[2]))
    dt = to_pose[2]-from_pose[2]
    return dx, dy, dt
    # return np.array([dx,dy,dt])

def rotated_relative_position(dx,dy,theta):
    dxR = dx*cos(theta)-dy*sin(theta)
    dyR = dx*sin(theta)+dy*cos(theta)
    return dxR, dyR

def plot_scan(scan1,ax,mycolor='b',label="none"):
    x_coords, y_coords = zip(*scan1)
    ax.scatter(x_coords,y_coords,color=mycolor,label=label)
    
def reregistration(scanlist,poselist,odom_poses):
    fig = plt.figure();
    ax = plt.subplot(111)
    for ind_one in range(0,len(scanlist)):
        transformed_scan=scan_match.transform_scan(scanlist[ind_one],odom_poses[ind_one][0],odom_poses[ind_one][1],odom_poses[ind_one][2])
        # transformed_scan=scan_match.transform_scan(scanlist[ind_one],poselist[ind_one][0],poselist[ind_one][1],poselist[ind_one][2])
        print(poselist[ind_one],odom_poses[ind_one])
        plot_scan(transformed_scan,ax)
    plt.show()
    transformed_scan_1 = scan_match.transform_scan(scanlist[0],odom_poses[0][0],odom_poses[0][1],odom_poses[0][2])
    for i in range(1,len(scanlist)):
        transformed_scan_i=scan_match.transform_scan(scanlist[i],odom_poses[i][0],odom_poses[i][1],odom_poses[i][2])
        match_result, match_cov =scan_match.ndt_scan_match_hp(transformed_scan_i,transformed_scan_1,0.5,max_iters=500) 
        dx_match = match_result["translation"][0]
        dy_match = match_result["translation"][1]
        dt_match = match_result["rotation"]
        # print("mtch res:",dx_match,dy_match,dt_match)  
        dx,dy,dt = relative_transform(odom_poses[i],poselist[i])
        # print("real rel:",dx,dy,dt)
        dxR, dyR, dtR = invert_transform(np.array([dx_match,dy_match,dt_match]))
        # print("real rel:",dx,dy,dt)
        dx,dy,dt = relative_transform(poselist[i],odom_poses[i])
        # print("est  rel:",dxR,dyR,dtR)
        # print("pose:",poselist[i],"odom:",odom_poses[i],"match:",dx_match,dy_match,dt_match)
        # fig = plt.figure();
        # ax = plt.subplot(111)
        # plot_scan(transformed_scan_1,ax,mycolor='r')
        # plot_scan(transformed_scan_i,ax,mycolor='b')
        # transformed_scan_i_match=scan_match.transform_scan(scanlist[i],odom_poses[i][0]+dx_match,odom_poses[i][1],odom_poses[i][2])
        # plot_scan(transformed_scan_i_match,ax,mycolor='c')
        # plt.show(block=False)
        # plt.pause(0.1)
        # plt.close()
        ## Correct the orientation only:
        odom_poses[i][2]+=dtR

    fig = plt.figure();
    ax = plt.subplot(111)
    for ind_one in range(0,len(scanlist)):
        transformed_scan=scan_match.transform_scan(scanlist[ind_one],odom_poses[ind_one][0],odom_poses[ind_one][1],odom_poses[ind_one][2])
        # transformed_scan=scan_match.transform_scan(scanlist[ind_one],poselist[ind_one][0],poselist[ind_one][1],poselist[ind_one][2])
        print(poselist[ind_one],odom_poses[ind_one])
        plot_scan(transformed_scan,ax)
        ax.scatter(odom_poses[ind_one][0],odom_poses[ind_one][1])
    plt.show()

    np.savez(args.prefix+str(args.fnum)+"odom_corrected"+args.suffix, *odom_poses)
    return odom_poses



def mapping(scanlist,poselist,odom_poses):
    n = len(scanlist)
    print("number of scans:", n)
    # Define the edges as pairs picked from scans
    old_odom_poses=copy.deepcopy(odom_poses)
    G=[]
    first_ind = 0
    final_ind = n
    ind_interval = 4
    for i in range(first_ind,final_ind-2*ind_interval,ind_interval):
        G.append([i,i+ind_interval])
        G.append([i,i+2*ind_interval])
    first_ind = 0
    final_ind = n
    ind_interval = 1
    for i in range(first_ind,final_ind-2*ind_interval,ind_interval):
        G.append([i,i+ind_interval])
        G.append([i,i+2*ind_interval])
    # for i in range(first_ind,final_ind-6*ind_interval,ind_interval):
    #     G.append([i,i+6*ind_interval])
    set1 = list(range(first_ind,final_ind,ind_interval))
    ## Manual: Make sure the nodes added in edges below are in set1, for the vertex_dict to work
    # G=[]
    G.append([0,n-1])
    G.append([0,60])
    G.append([0,40])
    G.append([40,60])
    G.append([40,120])
    G.append([60,120])
    if 40 not in set1:
        set1.append(40)
    if 60 not in set1:
        set1.append(60)
    if 120 not in set1:
        set1.append(120)
    # set1 = [0,40,60,120]
    set2 = range(0,len(set1))

    # using a dictionary comprehension
    vertex_dict = {k: v for k, v in zip(set1, set2)}

    slam_pairwise_hessian=[]
    slam_pairwise_relpose=[]
    manual_poses = np.zeros((10,3))
    ## Pairwise scan matching for edges in graph G
    for g in G:
        print("\nEdge: ",g)
        ind_one = g[0]
        ind_two = g[1]
        print(odom_poses[ind_one],odom_poses[ind_two])

        dx,dy,dt = relative_transform(poselist[ind_one],poselist[ind_two])
        print("real rel in F1\t\t:",dx,dy,dt)
        dx,dy,dt = relative_transform(odom_poses[ind_one],odom_poses[ind_two])
        print("old odom rel in F1:\t",dx,dy,dt)
        ## We are initializing based on odometry
        dx_inv,dy_inv,dt_inv = invert_transform(np.array([dx,dy,dt]))
        print("old odom rel in F2:",dx_inv,dy_inv,dt_inv)
        match_result, match_cov =scan_match.ndt_scan_match_hp(scanlist[ind_two],scanlist[ind_one],2.0,tx_init=dx_inv,ty_init=dy_inv,phi_init=dt_inv,max_iters=500) ## find parameters needed to push prev scan to current scan, use  odometry to init
        manual_poses[0][0] = match_result["translation"][0]
        manual_poses[0][1] = match_result["translation"][1]
        manual_poses[0][2] = match_result["rotation"]
        print("new odom rel in F2:",manual_poses[0][0],manual_poses[0][1],manual_poses[0][2])
        print("match result should match odom rel in F2")
        # print("match cov:", match_cov)
        # dxR, dyR = rotated_relative_position( dx,dy,poselist[ind_one][2])
        # print("real rel:",dxR,dyR)

        dx_inv,dy_inv,dt_inv = invert_transform(manual_poses[0])
        print("new odom rel in F1:",dx_inv,dy_inv,dt_inv)
        dxR, dyR = rotated_relative_position(dx_inv,dy_inv,odom_poses[ind_one][2])
        print("new est rel in world:",dxR,dyR)
        print("real rel in world:\t\t", poselist[ind_two]-poselist[ind_one])
        print("old est rel in world:\t\t", odom_poses[ind_two]-odom_poses[ind_one])

        # dxR, dyR, dtR = invert_transform(manual_poses[0])
        # print("est  rel:",dxR,dyR,dtR)

        slam_pairwise_relpose.append(np.array([dxR,dyR,-manual_poses[0][2]]))
        # slam_pairwise_relpose.append(np.array([dxR,dyR,dtR]))
        # slam_pairwise_relpose.append(odom_poses[ind_two]-odom_poses[ind_one])

        # transformed_scan=scan_match.transform_scan(scanlist[ind_one],manual_poses[0][0],manual_poses[0][1],manual_poses[0][2])
        # fig = plt.figure();
        # ax = plt.subplot(111)
        # ndt_grid = scan_match.compute_ndt_grid(scanlist[ind_two], 2.0)
        # for g in ndt_grid:
        #     scan_match.confidence_ellipse(ndt_grid[g][0],ndt_grid[g][1] , ax,3,alpha=0.5, facecolor='pink')
        #     ax.scatter(ndt_grid[g][0][0],ndt_grid[g][0][1],color='y')
        # x1, y1 = zip(*scanlist[ind_one])
        # x2, y2 = zip(*scanlist[ind_two])
        # x3, y3 = zip(*transformed_scan)
        # ax.scatter(x1,y1,color='r',label="first scan")
        # ax.scatter(x2,y2,color='b',label="second scan")
        # ax.scatter(x3,y3,color='c',label="first transformed")
        # plt.show(block=False)
        # plt.pause(0.2)
        # plt.close()

    # V = range(0,n-5,5) ## vertices. G are the edges
    # for v in V:
    #     print(poselist[v])

    for l in range(0,len(slam_pairwise_relpose)):
        print("estimate rel :", slam_pairwise_relpose[l])
        print("true relative:   ", poselist[G[l][1]]-poselist[G[l][0]])
        print("poses        :   ", poselist[G[l][0]],poselist[G[l][1]])


    nedges = len(G)
    A_mat = np.zeros((3*nedges+3,3*len(set2)))
    b_vec = np.zeros(3*len(G)+3)
    for i in range(0,len(G)): ## Why are these on relative poses? They should be on absolute, no?
        print(i, G[i])
        b_vec[3*i] = slam_pairwise_relpose[i][0]
        b_vec[3*i+1] = slam_pairwise_relpose[i][1]
        b_vec[3*i+2] = slam_pairwise_relpose[i][2]
        A_mat[3*i][3*vertex_dict[G[i][1]]] = 1
        A_mat[3*i][3*vertex_dict[G[i][0]]] = -1
        A_mat[3*i+1][3*vertex_dict[G[i][1]]+1] = 1
        A_mat[3*i+1][3*vertex_dict[G[i][0]]+1] = -1
        A_mat[3*i+2][3*vertex_dict[G[i][1]]+2] = 1
        A_mat[3*i+2][3*vertex_dict[G[i][0]]+2] = -1
        # A_mat[3*i+1][vertex_dict[G[i][1]]] =
        # A_mat[3*i+2] =
    # A_mat+=0.001*np.identity(len(b_vec))
    A_mat[3*(nedges)][0] = 1.0
    A_mat[3*(nedges)+1][1] = 1.0
    A_mat[3*(nedges)+2][2] = 1.0

    b_vec[3*(nedges)] = 0.0
    b_vec[3*(nedges)+1] = 7.0
    b_vec[3*(nedges)+2] = 0.0
    sol = np.linalg.solve(A_mat.T @ A_mat,A_mat.T @ b_vec)

    # for i in range(1500):
    #     loss_value = 0.0
    #     gradvec = np.zeros((len(odom_poses),3))

    #     for g, l in zip(G,slam_pairwise_relpose):
    #         gradvec[g[0]]+= l-odom_poses[g[1]]+odom_poses[g[0]]
    #         gradvec[g[1]]-= l-odom_poses[g[1]]+odom_poses[g[0]]
        
    #     for p,g in zip(odom_poses,gradvec):
    #         p+=-0.1*g

    #     loss_value = 0.0
    #     for g, l in zip(G,slam_pairwise_relpose):
    #         loss_value+=np.linalg.norm(l -odom_poses[g[1]]+odom_poses[g[0]] )
    #     # print(loss_value)

    for v in set1:
        odom_poses[v][0] = sol[3*vertex_dict[v]]
        odom_poses[v][1] = sol[3*vertex_dict[v]+1]
        odom_poses[v][2] = sol[3*vertex_dict[v]+2]
        print(v,odom_poses[v],old_odom_poses[v],poselist[v])
    return odom_poses

    # for g,l in G:




def load_odom(args):
    ## loads the stored scans and their true poses and odometry-estimated poses, possible calculated from `odom`
    fname = args.prefix+str(args.fnum)+"_scans"+args.suffix
    data = np.load(fname)
    scanlist = [data[k] for k in data]

    fname = args.prefix+str(args.fnum)+"_poses"+args.suffix
    data = np.load(fname)
    poselist = [data[k] for k in data] # true poses from sim

    fname = args.prefix+str(args.fnum)+"_odom"+args.suffix
    data = np.load(fname)
    odom_poses = [data[k] for k in data]
    return scanlist, poselist, odom_poses


def main(args) -> None:
    scanlist, poselist, odom_poses = load_odom(args) 
    n = len(scanlist)
    print("number of scans:", n)
    # for p,o in zip(poselist,odom_poses):
    #     print(o-p)
    # fig = plt.figure();
    # ax = plt.subplot(111)

    # plt.show()
    ## Plot the scans superimposed in a common reference frame based on estimated poses where each scan was taken
    # for g in [80,90,100,110]:
    # for g in range(0,120,5):
    #     transformed_scan=scan_match.transform_scan(scanlist[g],poselist[g][0],poselist[g][1],-poselist[g][2]) ## this was better than dxi,dyi,dti
    #     x, y = zip(*transformed_scan) 
    #     ax.scatter(x,y,color='r')
    # plt.show()
    # we should get a 'sharper' map after we solve the correction optimization

    # x_coords, y_coords, t_coords = zip(*poselist)
    # ax.scatter(x_coords,y_coords,color='r')
    # x_coords, y_coords, t_coords = zip(*odom_poses)
    # ax.scatter(x_coords,y_coords,color='b')
    print("done loading/computing odom")

    # odom_poses = reregistration(scanlist,poselist,odom_poses)
    odom_poses = mapping(scanlist,poselist,odom_poses)
    v0 = odom_poses[0]
    fig = plt.figure();
    ax = plt.subplot(111)

    for a,b in zip(poselist,odom_poses):
        ax.scatter(a[0],a[1],color='purple')
        ax.scatter(b[0],b[1],color='y')

    first_ind = 0
    final_ind = n
    ind_interval = 1
    set1 = range(first_ind,final_ind,ind_interval)
    # for a in odom_poses:
    for a in set1:
        ax.scatter(odom_poses[a][0]-v0[0],odom_poses[a][1]-v0[1]+7.0,color='c')
        ax.scatter(poselist[a][0],poselist[a][1],color='r')
    for scan,pose in zip(scanlist,odom_poses):
        transformed_scan=scan_match.transform_scan(scan,pose[0],pose[1],pose[2]) ## this was better than dxi,dyi,dti
        x, y = zip(*transformed_scan) 
        ax.scatter(x,y,color='y')
    fig.savefig(args.prefix+str(args.fnum)+'_slamplot.png')
    # plt.show()

    print("done")
    ## Plot the scans superimposed in a common reference frame based on estimated poses where each scan was taken
    # for g in range(0,120,5):
    #     transformed_scan=scan_match.transform_scan(scanlist[g],poselist[g][0],poselist[g][1],-poselist[g][2]) ## this was better than dxi,dyi,dti
    #     x, y = zip(*transformed_scan) 
    #     ax.scatter(x,y,color='y')
    # we should get a 'sharper' map after we solve the correction optimization

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process multiple files with sequential numbering")
    parser.add_argument("fnum", type=int, help="Starting integer for file sequence")
    parser.add_argument("--x0", type=float, default=0.0, help="Initial x (default=0.0")
    parser.add_argument("--y0", type=float, default=0.0, help="Initial y (default=0.0")
    parser.add_argument("--t0", type=float, default=0.0, help="Initial theta in deg (default=0.0")
    parser.add_argument("--prefix", type=str, default="simdata/sim_", help="Prefix for the files (default: 'simdata/sim')")
    parser.add_argument("--suffix", type=str, default=".npz", help="Suffix for the files (default: '.txt')")
    args = parser.parse_args()
    # print(args.fnum,args.prefix,args.suffix)
    main(args)
