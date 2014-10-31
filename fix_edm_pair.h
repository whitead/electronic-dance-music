#ifdef FIX_CLASS

FixStyle(edm_pair,FixEDMPair)

#else

#ifndef LMP_FIX_EDM_PAIR_H
#define LMP_FIX_EDM_PAIR_H

#include "fix.h"
#include "edm_bias.h"
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
#include "neigh_request.h"



namespace LAMMPS_NS {

class FixEDMPair : public Fix {
 public:
  FixEDMPair(class LAMMPS *, int, char **);
  ~FixEDMPair();
  // class EDMPair included: a pointer that initialize the class and create all the interfaces 
  class EDMPair *EDMPair;
  int setmask();
  void init();
  void setup(int);
  void min_setup(int);
  void post_force(int);
  void post_force_respa(int, int, int);
  void min_post_force(int);
  void init_list(int, class NeighList *); //override

 private:
   class EDMBias* bias;
   char bias_file[256];
   double temperature;
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

