#incldue "grid.h"

template<int DIM>
class GaussGrid : protected Grid<DIM> {
  /** A class for treating grids that have gaussians on it 
   *
   *
   **/
 public:
  GaussGrid(double* min, double* max, int* bin_number, int* b_periodic, double* sigma);
  void add_gaussian(double* x, double height);
  void set_boundary(double* min, double* max);
  double get_volume();
  
  
 private:
  double minisize[DIM];// On DIM-dimensional grid, how far we must search before gaussian decays enough to ignore
  double mini_onesize; //On reduced grid, how far we must search before gaussian decays enough to ignore
  double sigma[DIM];
  double boundary_min[DIM]; //optional boundary minmimum
  double boundary_max[DIM]; //optional boundary maximum
  
};
