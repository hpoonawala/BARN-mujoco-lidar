import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from math import exp, sin, cos, atan2, log
from scipy.linalg import sqrtm

def confidence_ellipse(mu, cov, ax, n_std=3.0,facecolor='None',**kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mu[0]

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mu[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def compute_ndt_grid(scan, grid_size):
    """
    Computes the Normal Distributions Transform (NDT) grid for a given scan.

    Parameters:
        scan (ndarray): Nx2 array of (x, y) points.
        grid_size (float): The size of each grid cell.

    Returns:
        ndt_grid (dict): A dictionary where keys are grid cell indices, and values
                         are tuples (mean, covariance) for the Gaussian distribution.
    """
    ndt_grid = {}

    # Compute grid indices for each point
    grid_indices = np.floor(scan / grid_size).astype(int)

    # Group points by grid cell
    for idx in np.unique(grid_indices, axis=0):
        cell_points = scan[(grid_indices == idx).all(axis=1)]
        if cell_points.shape[0] > 2:  # At least 2 points to compute covariance
            mean = np.mean(cell_points, axis=0)
            covariance = np.cov(cell_points.T)
            # Regularize covariance to avoid singular matrices
            covariance += np.eye(2) * 1e-6
            # eig1, eig2, v1, v2 = myeig(covariance)
            # if eig1/eig2+eig2/eig1 > 200:
            ndt_grid[tuple(idx)] = (mean, covariance)

    return ndt_grid

def transform_scan(scan, tx, ty, phi):
    """
    Applies a rigid-body transformation to a scan.

    Parameters:
        scan (ndarray): Nx2 array of (x, y) points.
        tx, ty (float): Translation parameters.
        phi (float): Rotation parameter (in radians).

    Returns:
        transformed_scan (ndarray): Transformed scan.
    """
    rotation = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi),  np.cos(phi)]])
    return (rotation @ scan.T).T + np.array([tx, ty])

def compute_ndt_score(scan, ndt_grid, grid_size):
    """
    Computes the NDT score for a transformed scan.

    Parameters:
        scan (ndarray): Nx2 array of (x, y) points.
        ndt_grid (dict): Precomputed NDT grid with Gaussian distributions.
        grid_size (float): Size of the grid cells.

    Returns:
        score (float): Negative log-likelihood score.
    """
    score = 0.0
    d1, d2, d3 = 1.4663370687934272, 0.5643202489410892, 1.2039728043259361
    grid_indices = np.floor(scan / grid_size).astype(int)

    for point, idx in zip(scan, grid_indices):
        cell_key = tuple(idx)
        if cell_key in ndt_grid:
            mean, covariance = ndt_grid[cell_key]
            v = point - mean # error of point from mean
            d = v @ np.linalg.inv(covariance) @ np.transpose(v)
            score += d3 - d1*exp(-0.5*d2*d)
            prob = multivariate_normal.pdf(point, mean=mean, cov=covariance)

    return score  # Negative log-likelihood

def myeig(K):
    trK = K[0][0]+K[1][1]
    detK = K[0][0]*K[1][1]-K[0][1]*K[1][0]
    eig1 = trK/2+np.sqrt(trK*trK-4*detK)/2
    eig2 = trK/2-np.sqrt(trK*trK-4*detK)/2
    v1 = np.array([K[0][0]-eig2,K[1][0]])
    v2 = np.array([K[0][1],K[1][1]-eig1])
    return eig1, eig2, v1, v2

def is_pd(K):
    trK = K[0][0]+K[1][1]
    detK = K[0][0]*K[1][1]-K[0][1]*K[1][0]
    eig1 = trK/2+np.sqrt(trK*trK-4*detK)/2
    eig2 = trK/2-np.sqrt(trK*trK-4*detK)/2
    if eig1 > 0 and eig2 > 0:
        return 1
    else:
        return 0

def ndt_scan_match_hp(scan2, scan1, grid_size, max_iters=250, tol=1e-6,tx_init=0.0,ty_init=0.0,phi_init=0.0):
    """
    Matches two 2D LiDAR scans using the Normal Distributions Transform (NDT).

    Parameters:
        scan1 (ndarray): Nx2 array of reference scan points (x, y).
        scan2 (ndarray): Nx2 array of points to align (x, y).
        grid_size (float): Size of the grid cells for NDT.
        max_iters (int): Maximum number of iterations.
        tol (float): Convergence tolerance.

    Returns:
        transform (dict): Transformation parameters {"translation": (tx, ty), "rotation": phi}.
    """
    # Compute NDT grid for the reference scan
    ndt_grid = compute_ndt_grid(scan2, grid_size)

    # Initialize parameters
    tx, ty, phi = tx_init,ty_init,phi_init

    count = 0
    tempsquare = np.zeros((3,3))
    d1, d2, d3 = 1.4663370687934272, 0.5643202489410892, 1.8971199848858813
    d1, d2, d3 = 1.4663370687934272, 0.5643202489410892, 1.2039728043259361
    # d1 = 1.0;d2=1.0;d3=0.0
    prev_score2 = 0.0
    # fig = plt.figure();
    # ax = plt.subplot(111)
    iterate_values = np.zeros(max_iters)
    for count in range(max_iters):
        transformed_scan = transform_scan(scan1, tx, ty, phi)
        Amat = np.zeros((3,3)) + 0.00 * np.diag([1.0,1.,1.0]) ## build the Hessian
        bvec = np.zeros(3) ## build the gradient

        grid_indices = np.floor(transformed_scan / grid_size).astype(int)
        score1 = 0.0
        score2 = 0.0
        
        for point, idx in zip(transformed_scan, grid_indices):
            cell_key = tuple(idx)
            if cell_key in ndt_grid:
                mean, covariance = ndt_grid[cell_key]
                prob = multivariate_normal.pdf(point, mean=mean, cov=covariance)
                if prob > 0:
                    score1 += -np.log(prob)
                Jp = np.array([[1,0,-point[0]*sin(phi)-point[1]*cos(phi)],[0,1,point[0]*cos(phi)-point[1]*sin(phi)] ] )
                v = point - mean # error of point from mean
                d = v @ np.linalg.inv(covariance) @ np.transpose(v)
                e = -d1*d2/2*exp(-0.5*d*d2)
                score2 += d3 - d1*exp(-0.5*d2*d)
                tempvec = np.transpose(Jp) @ np.linalg.inv(covariance) @ v # gradient of quadratic transpose: 3-vector
                bvec+= -e* tempvec
                Amat+= -e*np.transpose(Jp) @ np.linalg.inv(covariance) @ Jp 
                Amat[2][2]+= -e*(-v[0]*Jp[1][2] + v[1]*Jp[0][2])
                for i in range(3):
                    for j in range(3):
                        tempsquare[i][j] = tempvec[i]*tempvec[j]
                Amat+=  e*d2/2 *tempsquare 
        if score2 > 0: 
            iterate_values[count] = log(score2)
        ## Can add regularization here 
        ## Is isnotposdef(Amat)
        ## Amat += beta*Identity (beta = 0.5?)
        while is_pd(Amat) == 0:
            Amat+=0.5*np.identity(3)

        # print(  np.linalg.inv(Amat) @ bvec)
        sol = np.linalg.inv(Amat) @ bvec

        ## This next step is gradient clipping 
        # norm_sol = np.linalg.norm(sol)
        # if norm_sol >0.2:
        #     sol=sol/norm_sol*0.2
        # sol[2]=0.0
        # print(count, "score1: ",score1, "score2: ",score2)
        # print("step:",sol, "step norm:",np.linalg.norm(sol))
        # print("gradient:",bvec)

        # Compute gradients (numerical approximation)
        grad_tx = sol[0]
        grad_ty = sol[1]
        grad_phi = sol[2]


        ## Can use the ability to compute score to implement back-tracking line search here
        alpha = 1;
        c = 0.5;
        b = 0.01;

        score3 = compute_ndt_score(transformed_scan,ndt_grid,grid_indices)
        transformed_scan_test = transform_scan(scan1, tx-alpha*grad_tx, ty-alpha*grad_ty, phi-alpha*grad_phi)
        score_next = compute_ndt_score(transformed_scan_test,ndt_grid,grid_indices)
        flag = score_next > score3 - b*alpha*(bvec.T @ sol) ## bvec.T @ sol is grad^T Hessian^{-1} grad, which is lambda^2. One criterion for stopping is lambda^2 <= 2 epsilon 
        # print(score_next,score3,b*alpha*bvec.T @ sol,flag) 
        # print(count," flag:",flag)
        while (flag):
            alpha=alpha*0.5
            transformed_scan_test = transform_scan(scan1, tx-alpha*grad_tx, ty-alpha*grad_ty, phi-alpha*grad_phi)
            score_next = compute_ndt_score(transformed_scan_test,ndt_grid,grid_indices)
            flag = score_next > score3 - b*alpha*(bvec.T @ sol) ## bvec.T @ sol is grad^T Hessian^{-1} grad, which is lambda^2. One criterion for stopping is lambda^2 <= 2 epsilon 
            # print(score_next,score3,b*alpha*bvec.T @ sol,flag) 

        # while (Armijo condition is satisfied, terms computed here): ## f(x0 + alpha* dX) > f(x0) + b*alpha*gradf(x0)*dX implies reduce alpha
        #     alpha = c*alpha;
        print(alpha)

        # Update parameters using gradient ascent
        tx -= grad_tx * alpha ## replace constant with alpha from back-tracking
        ty -= grad_ty *   alpha
        phi -= grad_phi * alpha
        # print(tx,ty,phi)
        
        # x_coords, y_coords = zip(*transformed_scan)
        # ax.scatter(x_coords, y_coords)
        # x_coords, y_coords = zip(*scan2)
        # ax.scatter(x_coords, y_coords,color='b')
        # plt.show()
        # fig = plt.figure();
        # ax = plt.subplot(111)
        # for p1,p2,p3 in zip(scan1,scan2,transformed_scan):
        #     ax.scatter(p1[0],p1[1],color='r')
        #     ax.scatter(p2[0],p2[1],color='b')
        #     ax.scatter(p3[0],p3[1],color='c')

        # for point, idx in zip(transformed_scan, grid_indices):
        #     cell_key = tuple(idx)
        #     if cell_key in ndt_grid:
        #         mean, covariance = ndt_grid[cell_key]
        #         ax.plot([point[0],mean[0]],[point[1],mean[1]])
        # Check for convergence
        # print(bvec)
        if np.linalg.norm(bvec) < tol:
            break
        if count > 3 and abs(score2-prev_score2)<1e-4:
            break
        prev_score2 = score2;
            

    # fig = plt.figure();
    # ax = plt.subplot(111)
    # plt.scatter(range(count),iterate_values[:count])
    # plt.yscale('log')
    # plt.show()
    # plt.show()
    # print("newton iterations: ",count)
        
    return {"translation": (tx, ty), "rotation": phi}, Amat



# Example usage
if __name__ == "__main__":
    # Example LiDAR scans (Nx2 arrays of (x, y) points)
    scan1a = (np.random.rand(100, 2) @ np.diag([0.1,5])) + np.array([2.1,0])  # Reference scan
    scan1b = (np.random.rand(100, 2) @ np.diag([0.1,5])) - np.array([2.1,0])  # Reference scan
    scan1c = (np.random.rand(100, 2) @ np.diag([5,0.1])) + np.array([0,5])  # Reference scan
    scan1=np.vstack((scan1a,scan1b,scan1c))
    print(scan1.shape)
    tx_init = 8.05
    ty_init = 7.95
    phi_init = 0.85
    scan4 = transform_scan(scan1, tx=tx_init, ty=ty_init, phi=phi_init)
    tx_true = 8.0
    ty_true = 8.0
    phi_true = 0.8
    scan2 = transform_scan(scan1, tx=tx_true, ty=ty_true, phi=phi_true) + np.random.normal(0, 0.01, scan1.shape)

    # Perform scan matching
    grid_size = 2.5  # Grid size for NDT
    # result = ndt_scan_match(scan1, scan2, grid_size) # original
    # result = ndt_scan_match_oneshot(scan1, scan2, grid_size) # sin, cos params
    result = ndt_scan_match_hp(scan2, scan1, grid_size,max_iters=350,tol=10,tx_init=tx_init,ty_init=ty_init,phi_init=phi_init) # my nonlin implementation
    print("Estimated Transformation:", result)
    print("ground truth: ",tx_true,ty_true,phi_true)
    a, b = result["translation"]

    scan3 = transform_scan(scan1, tx=a, ty=b, phi=result["rotation"]) 
    grid_indices = np.floor(scan2 / grid_size).astype(int)
    ndt_grid = compute_ndt_grid(scan1, grid_size)
    print(compute_ndt_score(scan1,ndt_grid,grid_indices))
    grid_indices = np.floor(scan2 / grid_size).astype(int)
    print(compute_ndt_score(scan2,ndt_grid,grid_indices))
    grid_indices = np.floor(scan3 / grid_size).astype(int)
    print(compute_ndt_score(scan3,ndt_grid,grid_indices))

    fig = plt.figure();
    ax = plt.subplot(111)
    grid_indices = np.floor(scan2 / grid_size).astype(int)
    for g in ndt_grid:
        confidence_ellipse(ndt_grid[g][0],ndt_grid[g][1] , ax,3,alpha=0.5, facecolor='pink')
        ax.scatter(ndt_grid[g][0][0],ndt_grid[g][0][1],color='y')
    for p1,p2,p3,p4 in zip(scan1,scan2,scan3,scan4):
        ax.scatter(p1[0],p1[1],color='r')
        ax.scatter(p2[0],p2[1],color='b')
        ax.scatter(p4[0],p4[1],color='y')
        ax.scatter(p3[0],p3[1],color='c')
    for p in grid_indices:
        ax.scatter(p[0]*grid_size,p[1]*grid_size,color='g')
    plt.show()

