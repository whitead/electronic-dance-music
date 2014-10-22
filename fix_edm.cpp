/**
 * Example syntax:
 *   fix ID group-ID EDM input_file
 *
 **/

#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "atom.h"
#include "update.h"
#include "force.h"
#include "respa.h"
#include "domain.h"
#include "error.h"
#include "group.h"
#include "fix_edm.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

FixEDM::FixEDM(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  double *mss,*chg;
  int *int_p,size;
  int i,j,k,i_c,nn,mm;
  int me;

  if (narg != 5) error->all(FLERR,"Illegal fix EDM command");

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&size);

  if (!atom->tag_enable)
    error->all(FLERR,"fix EDM requires atom tags");

  //here is where we would load up the EDM bias

  return;
}

/* ---------------------------------------------------------------------- */

FixEDM::~FixEDM()
{

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
  //update biases
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
