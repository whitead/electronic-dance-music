electronic-dance-music
======================

Experiment directed metadynamics plugin in for lammps or other
possibly future engines. 


TODO
===============

1. Implement distinction between system boundarie and bias boundaries  Change Subdivide Method
2. Refactor edm_bias into edm_coord and edm_pair

edm_bias
=======

Methods to refactor
------
1. subdivide
2. add_hills (pair-wise vs position)
3. infer_neighbors/sort_neighbors
4. read_input --> outside of edm hiearchy 
5. update_forces

Variables to refactor
------
None

add_hills
-----
Change to take a double that is the CV, not positions. I think that's it there

subidivide
------
To subclass

infer_neighbors/sort_neighbors
------
Virtual, do nothing for pair except say all are neighbors

read_input
------
Needs to be static, separate method defined in edm_bias.h. 
Should take enum declaring which edm_bias type to instantiate

update_force
------
Same as add_hills, just take CV instead of positions. 

fix_edm
=====

If pair, must calculate and pass pairwise distances instead of
positions to force and hill calculations.

