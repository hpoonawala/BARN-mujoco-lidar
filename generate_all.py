import subprocess

for i in range(10):
    subprocess.run(['python3', 'generate_plots.py', str(i),"--prefix=simdata/sim_"])
