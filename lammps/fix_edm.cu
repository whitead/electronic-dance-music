/**
 * Example syntax:
 *   fix [ID] [group-ID] edm [temperature] [input_file] [add hill stride, integer] [write bias stride, integer] [bias file] [seed]
 *
 * make sure write bias is large because that takes a long time
 **/

#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "atom.h"
#include "update.h"
#include "force.h"
#include "respa.h"
#include "domain.h"
#include "random_mars.h"
#include "error.h"
#include "group.h"
#include "memory.h"
#include "fix_edm.h"
#include "neighbor.h"
#ifdef __CUDACC__
#include "edm_bias_gpu.cuh"
#endif

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

FixEDM::FixEDM(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{

  int me, size;

  if (narg < 9) error->all(FLERR,"Illegal fix EDM command");

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&size);

  if (!atom->tag_enable)
    error->all(FLERR,"fix EDM requires atom tags");

  temperature = atof(arg[3]);
  stride = atoi(arg[5]);
  write_stride = atoi(arg[6]);
  strcpy(bias_file, arg[7]);
  seed = atoi(arg[8]);
  if(stride < 0)
    error->all(FLERR,"Illegal stride given to EDM command");
  if(write_stride < 0)
    error->all(FLERR,"Illegal write bias stride given to EDM command");


  //here is where we would load up the EDM bias
  #ifdef __CUDACC__
  bias = new EDM::EDMBiasGPU(arg[4]);
  #else
  bias = new EDM::EDMBias(arg[4]);
  #endif

  //indicate we calculate energy
  thermo_energy = 1;

  random_numbers = NULL;
  random = new RanMars(lmp,seed + me);

  return;
}

/* ---------------------------------------------------------------------- */

FixEDM::~FixEDM()
{
  if(bias != NULL)
    delete bias;
  if(bias != NULL)
    delete random;
  if(edm_x != NULL)
    delete edm_x;
  if(edm_f != NULL)
    delete edm_f;
  if(edm_random != NULL)
    delete edm_random;
  
  memory->destroy(random_numbers);
}

/* ---------------------------------------------------------------------- */

using namespace FixConst;
int FixEDM::setmask()
{
  // set with a bitmask how and when apply the force from EDM 
  int mask = 0;
  mask |= POST_FORCE;
  mask |= THERMO_ENERGY;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixEDM::init()
{

  if (strcmp(update->integrate_style,"respa") == 0)
    nlevels_respa = ((Respa *) update->integrate)->nlevels;

  bias->setup(temperature, force->boltz);
  //have to explicitly cast all these arrays to edm_data_t
  // can't automatically cast, apparently
  edm_data_t skin[3];
  edm_data_t sub_low[3];
  edm_data_t sub_high[3];
  edm_data_t box_low[3];
  edm_data_t box_high[3];
  for(int i = 0; i < 3; i++){
    sub_low[i] = (edm_data_t)domain->sublo[i];
    sub_high[i] = (edm_data_t)domain->subhi[i];
    box_low[i] = (edm_data_t)domain->boxlo[i];
    box_high[i] = (edm_data_t)domain->boxhi[i];
  }
  skin[0] = skin[1] = skin[2] = neighbor->skin;
  bias->subdivide(sub_low, sub_high, box_low, box_high, domain->periodicity, skin);
  bias->set_mask(atom->mask);

  //set energy just in case
  edm_energy = 0;
  num_atoms = atom->natoms;
  //have to malloc these arrays separately for the case of EDM using floats on GPU
  edm_x = (edm_data_t**) malloc(num_atoms*sizeof(edm_data_t*));
  for(int i = 0; i < num_atoms; i++){
    edm_x[i] = (edm_data_t*) malloc(3 * sizeof(edm_data_t));
  }
  edm_f = (edm_data_t**) malloc(num_atoms*sizeof(edm_data_t*));
  for(int i = 0; i < num_atoms; i++){
    edm_f[i] = (edm_data_t*) malloc(3 * sizeof(edm_data_t));
  }
  edm_random = (edm_data_t*) malloc(num_atoms * sizeof(edm_data_t));

}

/* ---------------------------------------------------------------------- */

void FixEDM::setup(int vflag)
{
  if (strcmp(update->integrate_style,"verlet") == 0)
    post_force(vflag);
  else {
    ((Respa *) update->integrate)->copy_flevel_f(nlevels_respa-1);
    post_force_respa(vflag,nlevels_respa-1,0);
    ((Respa *) update->integrate)->copy_f_flevel(nlevels_respa-1);
  }
}

/* ---------------------------------------------------------------------- */

void FixEDM::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixEDM::post_force(int vflag)
{

  //  domain->pbc(); //make sure particles are with thier processors
  //update the edm_data_t arrs
  int i, j;
  for(i = 0; i < atom->nlocal; i++) {
    for(j = 0; j< 3; j++){
      edm_x[i][j] = (edm_data_t) atom->x[i][j];
      edm_f[i][j] = (edm_data_t) atom->f[i][j];
    }
  }
  //update force/energy
  edm_energy = (double)(bias->update_forces(atom->nlocal, edm_x, edm_f, groupbit));
  //treat add hills
  if(update->ntimestep % stride == 0) {

    //bias requires payment in the form of an array of random numbers
    if(random_numbers == NULL) {
      memory->create(random_numbers, atom->nmax, "fix/edm:random_numbers");
    }
    for(i = 0; i < atom->nlocal; i++) {
      random_numbers[i] = random->uniform();
      edm_random[i] = (edm_data_t) random_numbers[i];
    }
    bias->add_hills(atom->nlocal, edm_x, edm_random, groupbit);
  }

  if(update->ntimestep % write_stride == 0) {
    bias->write_bias(bias_file);
    bias->write_histogram();
    bias->clear_histogram();    
  }

}

/* ---------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

void FixEDM::post_force_respa(int vflag, int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixEDM::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   Passing energy is apparently done via computer scalar...? I wish
   there was some documentation for this stuff.
 ---------------------------------------------------------------------- */
double FixEDM::compute_scalar() {
  return edm_energy;
}
