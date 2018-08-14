Electronic Dance Music
======================

Experiment directed metadynamics plugin for lammps or someday other
simulation engines. Experiment directed metadynamics is a method
to morph a molecular dynamics simulation to follow a desired free
energy surface (probability distribution). See this
[paper](http://pubs.acs.org/doi/abs/10.1021/acs.jctc.5b00178) This
plugin allows morphing pairwise interactions and coordinate
dimensions. These two collective variables are CPU intensive and
difficult on large systems, so electronic-dance-music uses a scalable
MPI-based approach to this.

What is unique to this implementation of metadynamics is that it
allows bias limiting. Generally in metadynamics, the bias added to a
system is a small amount. However, when biasing pairwise interactions
or coordinates of atoms there are many hills that can be added per
bias update step. Even if you choose to only add some, a large amount
of bias may be added. Electronic dance music solves this problem by
limiting the amount of energy added to the system per bias update. You
can't just stop adding hills after a certain amount of bias has been
added in a step, since that creates sample bias. So instead the hills that
should be added, but would add too much energy, are buffered and added
later.

You can choose how many hills to add per bias update, a desired amount
of energy to add to the bias per update step, and an upper limit to
the amount of bias that can be added.

Install
===

It only works with lammps right now and installing it is a little
roundabout, and only tested for Ubuntu. We're working on it. To install:

1. `mkdir buid && cd build`
2. `cmake .. && make`
3. Copy the compiled library in the `out` dir that has been created to somewhere a linker will easily find it, e.g. /usr/lib
4. Copy all files from `lammps` directory into the lammps src directory
5. Copy the version of `Makefile.ubuntu` from `lammps/lammps_makefiles` into the lammps `src/MAKE/MACHINES/`, replacing the one there. Also copy the `Makefile` from `lammps/lammps_makefiles` into the lammps `src` directory, replacing the one there. (This will compile with GPU support, but you can change the CMake flags if you want the CPU version.)
6. From lammps `src` directory, run `make-ubuntu`

Ignore all the CMake files floating aruond in the source, they're just
for unit tests.

Lammps Usage
====

Pairwise EDM
---

This is for running experiment directed metadynamics on pairs, for
example to match a radial distribution function. Please see the
[plumed_grids](https://github.com/whitead/plumed_grids) for converting
a radial distribution function into a probability distribution
function.

Pairwise EDM Fix
----

Your command should be:

    fix [fix-ID] [group-ID] edm_pair [temperature] [edm-input-file] [hill addition stride (integer)] [write bias stride (integer)] [edm input bias/target file] [random number seed] [rdf pairs]

With:

  * `fix-ID`: This is the ID for the fix that will be created
  * `group-ID`: The group of atoms on which the fix will be done. Should contain the types that are referred to in the `rdf pairs` argument
  * `temperature`: The temperature used for the bias. Almost always same as simulation temperature.  
  * `hill addition stride`: How many timesteps between hills adds
  * `write bias stride`: How often to write the bias. Writing the bias takes a long long time, so make this large. Each time this stride is reached, a lammps tabular potential is written out containing the current bias (edm_pair only) and a histogram of the collective variable is written.
  * `edm output bias file`: This is where the bias being applied will be written
  * `random number seed`: Some digits to seed the random number generator
  * `rdf pairs`: Just like the lammps rdf command, this should be either asterisks (match any type) or integer types.  Like `2 *` means match type 2 atoms in the given group-ID with anything.

The decision on what goes into the edm-input-file and what goes into the lammps fix were arbitrary and follows no logical reasons.

Coordinate EDM Fix
----

Everything is the same as above, except no rdf pair keyword needs to be added.

EDM Input File
====

Here's an example EDM input file demonstrating all options:

```C

//logical - indicates tempering or not
tempering		1

//If this is set to 0, then global tempering will be used. If this is
//non-zero, then thresholding will be used. If this is not set, local
//tempering is used
global_tempering	2.0

//hill height/prefactor
hill_prefactor		0.02 

//can control how much bias is added to the system in a given
//step. This is an upper limit which will not be exceeded. If this is
//not set, it's assumed to be the same as the hill prefactor.
bias_per_step 0.1

//This is not a bias restart, but the target that EDM will convert the
//PMF into. The file should contain your target as -ln(P(s)), where P(s)
//is the probability of the collective variable s
target_filename	       li_oc_target.dat

//The bias factor for well-tempering. Required if tempering is 1
bias_factor 		5.0

//The dimension of the bias. 
dimension 		1

//Hills added per step. 
hill_density		250

//In case you are targeting only a part of the PMF. For multiple
//dimensions use spaces between the items
box_low			1.68
box_high		5.0

//The grid spacing for bias calculations
bias_spacing		0.00025

//The gaussian standard deviation for the basis gaussians of the
//system
bias_sigma		0.025

//where to write hills
hills_filename		lioc_hills

//an initial bias to begin with 
initial_bias_filename   restart_bias.dat

//where to write a histogram of the CV.
//This is useful ensuring the target is reached and analyzing the PMF
//NOTE: The histogram is reset every time the bias file is rewritten.
//It is not a running histogram.

histogram_filename	cv_hist.dat

```

Boundaries
===

Electronic-dance-music uses the McGovern-De Pablo-boundary-corrected
and zero-force hills described in the original
[EDM paper](http://pubs.acs.org/doi/abs/10.1021/acs.jctc.5b00178) in 1
dimension. In more than 1, YOUR TARGET SHOULD ENCOMPASS THE WHOLE
REGION. Electronic-dance-music doesn't check for this, but you will
get CRAZY boundary behavior if you have a partial bias.


Grid Format
===

The grids should be written in Plumed 1 format and can be manipulated
using the [plumed_grids](https://github.com/whitead/plumed_grids) python plugin.

TODO
===

1. Add check for bad boundaries
2. Implement zero-force boundaries in multiple dimensions
3. Generalize edm_bias to > 3 dimensions (update_force and create subdivide alternative commands)

Citation
===

**Designing Free Energy Surfaces that Match Experimental Data with Metadynamics**. AD White, JF Dama, GA Voth. *J. Chem. Theory Comput.* **2015**, *11 (6)*, pp 2451-2460
