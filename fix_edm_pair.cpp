/**
 * Example syntax:
 *   fix [ID] [group-ID] edm [temperature] [input_file] [add hill stride, integer] [write bias stride, integer] [bias file] [seed] [rdf pairs]
 *
 * make sure write bias is large because that takes a long time
 **/

#include "fix_edm_pair.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

FixEDMPair::FixEDMPair(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{

  int me, size;

  if (narg < 11) error->all(FLERR,"Illegal fix edm_pair command");

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&size);

  if (!atom->tag_enable)
    error->all(FLERR,"fix edm_pair requires atom tags");

  temperature = atof(arg[3]);
  stride = atoi(arg[5]);
  write_stride = atoi(arg[6]);
  strcpy(bias_file, arg[7]);
  seed = atoi(arg[8]);
  if(stride < 0)
    error->all(FLERR,"Illegal stride given to edm_pair command");
  if(write_stride < 0)
    error->all(FLERR,"Illegal write bias stride given to edm_pair command");

  ipair = atoi(arg[9]);
  jpair = atoi(arg[10]);
  
  if(!ipair || !jpair) {
    error->all(FLERR, "Illegeal EDM command, invalid types");
  }

  //by default calculate energy
  thermo_energy = 1;

  //here is where we would load up the EDM bias
  bias = new EDM::EDMBias(arg[4]);

  if(bias->dim_ != 1)
    error->all(FLERR, "Pairwise distance must be 1 dimension in EDM input file");

  random = new RanMars(lmp,seed + me);

  return;
}

/* ---------------------------------------------------------------------- */

FixEDMPair::~FixEDMPair()
{
  if(bias != NULL)
    delete bias;
  if(bias != NULL)
    delete random;
  
}

/* ---------------------------------------------------------------------- */

using namespace FixConst;
int FixEDMPair::setmask()
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

void FixEDMPair::init()
{

  if (strcmp(update->integrate_style,"respa") == 0)
    nlevels_respa = ((Respa *) update->integrate)->nlevels;

  bias->setup(temperature, force->boltz);

  //The bounds for every node should be the bounds of the pairwise force
  double skin[3];
  skin[0] = neighbor->skin;
  double lo[3], hi[3];
  lo[0] = 0;
  // cannot use neighbor->cutneighmax b/c neighbor has not yet been init
  hi[0] = force->pair->cutforce + neighbor->skin;
  int p[3] = {0,0,0};

  bias->subdivide(lo, hi, lo, hi, p, skin);
  last_calls = atom->nmax; //make very conservative estimate of the number of pairs

  //request neighbor lists
  int irequest = neighbor->request((void *) this);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix = 1;


  //set energy just in case
  edm_energy = 0;
}

/* ---------------------------------------------------------------------- */

void FixEDMPair::setup(int vflag)
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

void FixEDMPair::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixEDMPair::post_force(int vflag)
{
  
  //  domain->pbc(); //make sure particles are with thier processors


  //neighbor list loop, ripped off of pair_lj_cut
  int i,ii,j, jj, inum, jnum, itype, jtype, ncalls = 0;
  double xtmp, ytmp, ztmp, delx, dely, delz, r;
  double edm_force[1];
  int* ilist, *jlist, *numneigh, **firstneigh;

  double **x = atom->x;
  double **f = atom->f;
  int* type = atom->type;
  int* mask = atom->mask;
  int newton_pair = force->newton_pair;
  
  int nlocal = atom->nlocal;
  if(newton_pair) {
    error->all(FLERR,"fix edm_pair requires 'newton off' to be declared in the lammps input script");
  }


  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;    
  
  edm_energy = 0;

  if(update->ntimestep % stride == 0)
    bias->pre_add_hill(last_calls);

  
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = type[i];
    if(itype != ipair)
      continue;

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK; //no idea why....
      jtype = type[j];

      if(jtype != jpair)
	continue;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      r = sqrt(delx*delx + dely*dely + delz*delz);      
      
      //get force on r-vector
      edm_force[0] = 0;
      edm_energy += bias->update_force(&r, edm_force);

      //convert to pair-wise force
      f[i][0] += delx * edm_force[0];
      f[i][1] += dely * edm_force[0];
      f[i][2] += delz * edm_force[0];
      if (newton_pair || j < nlocal) { //if we're not communicating or if j is local
       f[j][0] -= delx * edm_force[0];
       f[j][1] -= dely * edm_force[0];
       f[j][2] -= delz * edm_force[0];
      }

      //add hill
      if(update->ntimestep % stride == 0) {
	//make sure we don't double count
	if(atom->tag[i] < atom->tag[j]) {
	  bias->add_hill(last_calls, &r, random->uniform());
	  ncalls++;
	}
      }
      
    }
  }
  

  //store our calls for next time so we can add the correct number of hills
  if(update->ntimestep % stride == 0) {
    last_calls = ncalls;
    bias->post_add_hill();
  }
  
  if(update->ntimestep % write_stride == 0) {
    bias->write_bias(bias_file);
  }
 
}

/* ---------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

void FixEDMPair::post_force_respa(int vflag, int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixEDMPair::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   I have no idea what this is for, but it's how pair_lj_cut gets a neighbor
   list
 ---------------------------------------------------------------------- */
 void FixEDMPair::init_list(int id, NeighList *ptr) {

   list = ptr;
 }

/* ----------------------------------------------------------------------
   Passing energy is apparently done via computer scalar...? I wish
   there was some documentation for this stuff.
 ---------------------------------------------------------------------- */
double FixEDMPair::compute_scalar() {
  return edm_energy;
}
