#include "gaussian_grid.h"
#include "grid.h"
#include <string>

class EDMBias {
  /** The EDM bias class.
   *
   *
   */
 public:

  EDMBias(std::string& input_filename);
  /** Create a grid that only occupies enough space for this processes local box
   *
   */
  double subdivide(double sublo[3], double subhi[3]);
  void update(double temperature, int nlocal, int *ids, double **x, double **fexternal);
  
  
 private:
  int read_input(std::string& input_filename);
  void update_forces(int nlocal, double **positions,  double** fexternal);
  void add_hills(double temperature, int nlocal, double **positions);
  /** This will update the height, optionally with tempering. It also
   * will reduce across all processes the average height so that we
   * know if global tempering is necessary.
   *
   */
  void update_height(double bias_added);


  int b_tempering;// boolean, do tempering
  double global_tempering;// global tempering threshold
  double bias_factor;
  double boltzmann_factor;
  double hill_prefactor; //hill height prefactor
  double hill_density;// hills sampling density
  double cum_bias;//the current average bias  
  double total_volume;//total volume of grid 

  Grid& target; //target PMF
  GaussGrid& bias;// bias
  

}
