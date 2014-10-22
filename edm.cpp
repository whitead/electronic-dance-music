#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "stdint.h"

#include "many2one.h"
#include "one2many.h"
#include "files.h"
#include "memory.h"
#include "error.h"

#define QUOTE_(x) #x
#define QUOTE(x) QUOTE_(x)

#include "lmppath.h"
#include QUOTE(LMPPATH/src/lammps.h)
#include QUOTE(LMPPATH/src/library.h)
#include QUOTE(LMPPATH/src/input.h)
#include QUOTE(LMPPATH/src/modify.h)
#include QUOTE(LMPPATH/src/fix.h)
#include QUOTE(LMPPATH/src/fix_external.h)

#include "lmp_fix.h"

using namespace LAMMPS_NS;

void edm_callback(void *, bigint, int, int *, double **, double **);

struct Info {
  int me;
  Memory *memory;
  LAMMPS *lmp;
};

/* ---------------------------------------------------------------------- */

int main(int narg, char **arg)
{
  int n;
  char str[128];

  // setup MPI

  MPI_Init(&narg,&arg);
  MPI_Comm comm = MPI_COMM_WORLD;

  int me,nprocs;
  MPI_Comm_rank(comm,&me);
  MPI_Comm_size(comm,&nprocs);

  Memory *memory = new Memory(comm);
  Error *error = new Error(comm);

  // command-line args

  if (narg != 4) error->all("Syntax: edm Niter in.lammps in.edm");

  int niter = atoi(arg[1]);
  n = strlen(arg[2]) + 1;
  char *lammps_input = new char[n];
  strcpy(lammps_input,arg[2]);
  n = strlen(arg[3]) + 1;
  char *edm_input = new char[n];
  strcpy(edm_input,arg[3]);

  // instantiate LAMMPS

  LAMMPS *lmp = new LAMMPS(0,NULL,MPI_COMM_WORLD);

  // create simulation in LAMMPS from in.lammps

  lmp->input->file(lammps_input);

  // make info avaiable to callback function

  Info info;
  info.me = me;
  info.memory = memory;
  info.lmp = lmp;
  info.edm_input = edm_input;

  // set callback to Quest inside fix external
  // this could also be done thru Python, using a ctypes callback

  int ifix = lmp->modify->find_fix("2");
  FixExternal *fix = (FixExternal *) lmp->modify->fix[ifix];
  fix->set_callback(edm_callback,&info);

  // run LAMMPS for Niter
  // each time it needs forces, it will invoke edm_callback

  sprintf(str,"run %d",niter);
  lmp->input->one(str);

  // clean up

  delete lmp;

  delete memory;
  delete error;

  delete [] lammps_input;
  delete [] edm_input;

  MPI_Finalize();
}

/* ----------------------------------------------------------------------
   callback to Edm with atom IDs and coords from each proc
   invoke Edm to compute forces, load them into f for LAMMPS to use
   f can be NULL if proc owns no atoms
------------------------------------------------------------------------- */

void edm_callback(void *ptr, bigint ntimestep,
		    int nlocal, int *id, double **x, double **f)
{
  int i,j;
  char str[128];

  Info *info = (Info *) ptr;

  // boxlines = LAMMPS box size converted into Edm lattice vectors

  char **boxlines = NULL;
  if (info->me == 0) {
    boxlines = new char*[3];
    for (i = 0; i < 3; i++) boxlines[i] = new char[128];
  }

  double boxxlo = *((double *) lammps_extract_global(info->lmp,"boxxlo"));
}
