#ifndef GRID_H_
#define GRID_H_


template<int DIM>
class Grid {
  /** A DIM-dimensional grid for storing things
   *
   */
 public:
  Grid(double* min, double* max, int* bin_number, int* b_periodic, int* b_derivatives);
  /** This will actually allocate the arrays and perform any sublcass initialization
   *
   */
  virtual void initialize();
  double get_value(double* x);
  int multi2one(int index[DIM]);
  void write(std::string& filename);
  void read(std::string& filename);

  double dx[DIM];//grid spacing
  double min[DIM];//grid minimum
  double max[DIM];//maximum
  int bin_number[DIM];//number of bins, equal to points in periodic and points+1 in non-periodic
  int b_periodic[DIM];//if a dimension is periodic
  double* grid;//the grid values
  double* grid_deriv;//derivatives  
  

};

#endif //GRID_H_
