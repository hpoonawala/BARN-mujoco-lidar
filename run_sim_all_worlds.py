import subprocess

for i in range(0,2):
    subprocess.run(['python3', 'sim_world.py', str(i),"--prefix=simdata/sim_","--t0=90","--nsteps=75000"])
    subprocess.run(['python3', 'generate_plot.py', str(i),"--prefix=simdata/sim_"])
