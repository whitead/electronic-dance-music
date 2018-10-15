#ifdef FIX_CLASS

FixStyle(edm,FixEDM)

#else

#ifndef LMP_FIX_EDM_H
#define LMP_FIX_EDM_H

#include "fix.h"
#include "edm_bias.h"
#include "random_mars.h"

namespace LAMMPS_NS {

class FixEDM : public Fix {
 public:
  FixEDM(class LAMMPS *, int, char **);
  ~FixEDM();
  int setmask();
  void init();
  void setup(int);
  void min_setup(int);
  void post_force(int);
  void post_force_respa(int, int, int);
  void min_post_force(int);
  double compute_scalar(); //gives the energy

 private:
  class EDM::EDMBias* bias;
   char bias_file[256];
   double temperature;
   double edm_energy;
   int stride;
   int write_stride;
   double* random_numbers;
   class RanMars *random;
   unsigned int seed;
   int nlevels_respa;
   int64_t num_atoms;
   edm_data_t** edm_x;
   edm_data_t** edm_f;
   edm_data_t* edm_random;
};

};

#endif
#endif

