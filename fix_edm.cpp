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

#include "edm_bias.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

FixEDM::FixEDM(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{

  int me, size;

  if (narg != 9) error->all(FLERR,"Illegal fix EDM command");

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
  bias = new EDMBias(arg[4]);

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
  double skin[3];
  skin[0] = skin[1] = skin[2] = neighbor->skin;
  bias->subdivide(domain->sublo, domain->subhi, domain->boxlo, domain->boxhi, domain->periodicity, skin);
  bias->set_mask(atom->mask);

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

  //update force
  bias->update_forces(atom->nlocal, atom->x, atom->f, groupbit);  
  //treat add hills
  if(update->ntimestep % stride == 0) {

    //bias requires payment in the form of an array of random numbers
    if(random_numbers == NULL) {
      memory->create(random_numbers, atom->nmax, "fix/edm:random_numbers");
    }
    int i;
    for(i = 0; i < atom->nlocal; i++) {
      random_numbers[i] = random->uniform();
    }

    bias->add_hills(atom->nlocal, atom->x, random_numbers, groupbit);
  }

  if(update->ntimestep % write_stride == 0) {
    bias->write_bias(bias_file);
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
