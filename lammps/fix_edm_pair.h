#ifdef FIX_CLASS

FixStyle(edm_pair,FixEDMPair)

#else

#ifndef LMP_FIX_EDM_PAIR_H
#define LMP_FIX_EDM_PAIR_H

#include "fix.h"
#include <edm/edm_bias.h>
#include "random_mars.h"
#include "neigh_list.h"
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
#include "neighbor.h"
#include "pair.h"
#include "neigh_request.h"
#include "lmptype.h"



namespace LAMMPS_NS {

class FixEDMPair : public Fix {
 public:
  FixEDMPair(class LAMMPS *, int, char **);
  ~FixEDMPair();
  int setmask();
  void init();
  void setup(int);
  void min_setup(int);
  void post_force(int);
  void post_force_respa(int, int, int);
  void min_post_force(int);
  void init_list(int, class NeighList *); 
  double compute_scalar(); //gives the energy

 private:
  class EDM::EDMBias* bias;
   char bias_file[512];
   char lammps_table_file[512];
   double temperature;
   double edm_energy;
   int stride;
   int write_stride;
   double* random_numbers;
   double* f_buffer;
   double* dist_buffer;
   class RanMars *random;
   class NeighList *list; // half neighbor list
   unsigned int seed;
   int nlevels_respa;
   int last_calls; //an estimate of the number of pairs considered on this processor
   int ipair;// the types for considering
   int jpair;
};

};

#endif
#endif

