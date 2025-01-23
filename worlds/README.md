- Automated conversion of the world files from the BARN dataset to xml compatible with MuJoCo 
- Makefile applies nvim commands to files through command line
    - `.world` $\to$ `.csv`
- Python file `worlds/parse_worlds.py` converts the CSV into MuJoCo-readable XML
    -  `.csv`$\to$ `.xml`

To save space, the worlds are not included in this repo. They can be downloaded from [(download zip file)](https://cs.gmu.edu/~xiao/Research/BARN/BARN_dataset.zip).

Macros were a temporary option. This one cleans up the cylinder model to focus on pose and radius of a cylinder:
```
j/pose
dd?model
p/radius
dd?pose
pjdat/link
dat
```
It made more sense to only keep lines with `pose` tag, since the radii were identical, removetags, then convert to csv.




