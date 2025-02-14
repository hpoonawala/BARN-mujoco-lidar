import subprocess

for i in [20,25,30]:
    subprocess.run(['python3', 'sim_world.py', str(i),"--prefix=simdata/sim_","--y0=7.0","--nsteps=10000"])
    subprocess.run(['python3', 'generate_plot.py', str(i),"--prefix=simdata/sim_"])
    subprocess.run(['python3', 'odometry.py', str(i),"--prefix=simdata/sim_"])
    subprocess.run(['python3', 'slam.py', str(i),"--prefix=simdata/sim_"])
