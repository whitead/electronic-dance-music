#ifndef EDM_BIAS_H_
#define EDM_BIAS_H_

#include "gaussian_grid.h"
#include "grid.h"
#include <string>
#include <iterator>
#include <sstream>
#include <iostream>
#include <map>

#define BIAS_CLAMP 10

class EDMBias {
  /** The EDM bias class.
   *
   *
   */
 public:

  EDMBias(const std::string& input_filename);
  ~EDMBias();
  /** Create a grid that only occupies enough space for this processes local box
   *
   */
  void subdivide(const double sublo[3], const double subhi[3], const int b_periodic[3]);

  void setup(double temperature, double boltzmann);
    
  int read_input(const std::string& input_filename);
  void update_forces(int nlocal, const double* const* positions, double** forces, int apply_mask) const;
  void update_forces(int nlocal, const double* const* positions,  double** forces) const;
  void set_mask(const int* mask); //set a mask
  /**
   * Add hills. A precomputed array (different each time) needs to be
   * passed if stochastic sampling is done (otherwise NULL).
   **/
  void add_hills(int nlocal, const double* const* positions, const double* runiform);
  void add_hills(int nlocal, const double* const* positions, const double* runiform, int apply_mask);

  void write_bias(const std::string& output) const;

  /** This will update the height, optionally with tempering. It also
   * will reduce across all processes the average height so that we
   * know if global tempering is necessary.
   *
   */
  void update_height(double bias_added);


  int b_tempering_;// boolean, do tempering
  unsigned int dim_;//the dimension
  double global_tempering_;// global tempering threshold
  double bias_factor_;
  double boltzmann_factor_;
  double temperature_;
  double hill_prefactor_; //hill height prefactor
  double hill_density_;// hills sampling density
  double cum_bias_;//the current average bias  
  double total_volume_;//total volume of grid 

  int b_outofbounds_; //true if this MPI instance will always be out of grid

  double* bias_dx_; //bias spacing
  double* bias_sigma_;
  double* min_; //boundary minimum
  double* max_; //boundary maximum
  const int* mask_;// a mask to use to exclude atoms
  


  Grid* target_; //target PMF
  GaussGrid* bias_;// bias
  

};

#endif // EDM_BIAS_H_
