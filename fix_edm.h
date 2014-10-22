#ifdef FIX_CLASS

FixStyle(edm,FixEDM)

#else

#ifndef LMP_FIX_EDM_H
#define LMP_FIX_EDM_H

#include "fix.h"

namespace LAMMPS_NS {

class FixEDM : public Fix {
 public:
  FixEDM(class LAMMPS *, int, char **);
  ~FixEDM();
  // class EDM included: a pointer that initialize the class and create all the interfaces 
  class EDM *EDM;
  int setmask();
  void init();
  void setup(int);
  void min_setup(int);
  void post_force(int);
  void post_force_respa(int, int, int);
  void min_post_force(int);

 private:
};

};

#endif
#endif

