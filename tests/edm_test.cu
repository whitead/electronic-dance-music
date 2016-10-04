#include "grid.cuh"
#include "edm_bias.cuh"
#include "gaussian_grid.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE EDM
#include <boost/test/unit_test.hpp>
#include <boost/timer.hpp>
#include <sstream>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <sys/time.h>


#define EPSILON 1e-10
#define QUOTE(name) #name
#define STR(macro) QUOTE(macro)
#define GRID_SRC std::string(STR(TEST_GRID_SRC))
#define EDM_SRC std::string(STR(TEST_EDM_SRC))

using namespace boost;
using namespace EDM;


BOOST_AUTO_TEST_CASE( grid_1d_sanity )
{
  /* Visual:
   * values:   0   1   2   3
   * grid:   |---|---|---|---|---
   *         0   1   2   3   4
   */
  double min[] = {0};
  double max[] = {10};
  double bin_spacing[] = {1};
  int periodic[] = {0};
  DimmedGrid<1> g (min, max, bin_spacing, periodic, 0, 0);

  BOOST_REQUIRE_EQUAL(g.grid_number_[0], 11);
  BOOST_REQUIRE_EQUAL(g.grid_size_, 11);

  size_t array[] = {5};
  size_t temp[1];
  g.one2multi(g.multi2one(array), temp);
  BOOST_REQUIRE_EQUAL(array[0], temp[0]);

  for(int i = 0; i < 10; i++)
    g.grid_[i] = i;

  double x[] = {3.5};
  BOOST_REQUIRE(pow(g.get_value(x) - 3, 2) < 0.000001);

  //try to break it
  x[0] = 0;
  g.get_value(x);

  x[0] = 10;
  g.get_value(x);

}

BOOST_AUTO_TEST_CASE( grid_3d_sanity )
{
  double min[] = {-2, -5, -3};
  double max[] = {125, 63, 78};
  double bin_spacing[] = {1.27, 1.36, 0.643};
  int periodic[] = {0, 1, 1};
  DimmedGrid<3> g (min, max, bin_spacing, periodic, 0, 0);

  BOOST_REQUIRE_EQUAL(g.grid_number_[0], 101);
  BOOST_REQUIRE_EQUAL(g.grid_number_[1], 50);
  BOOST_REQUIRE_EQUAL(g.grid_number_[2], 126);

  size_t array[3];
  size_t temp[3];
  for(int i = 0; i < g.grid_number_[0]; i++) {
    array[0] = i;
    for(int j = 0; j < g.grid_number_[1]; j++) {
      array[1] = j;
      for(int k = 0; k < g.grid_number_[2]; k++) {
	array[2] = k;

	//check to make sure the index conversion is correct
	g.one2multi(g.multi2one(array), temp);
	BOOST_REQUIRE_EQUAL(array[0], temp[0]);
	BOOST_REQUIRE_EQUAL(array[1], temp[1]);
	BOOST_REQUIRE_EQUAL(array[2], temp[2]);

	g.grid_[g.multi2one(array)] = g.multi2one(array);
      }
    }
  }

  double point[3];
  for(int i = 0; i < g.grid_number_[0]; i++) {
    point[0] = i * g.dx_[0] + g.min_[0] + EPSILON;
    array[0] = i;
    for(int j = 0; j < g.grid_number_[1]; j++) {
      point[1] = j * g.dx_[1] + g.min_[1] + EPSILON;
      array[1] = j;
      for(int k = 0; k < g.grid_number_[2]; k++) {
	point[2] = k * g.dx_[2] + g.min_[2] + EPSILON;
	array[2] = k;
	BOOST_REQUIRE(pow(g.get_value(point) - g.multi2one(array),2) < 0.0000001);
      }
    }
  }  
}

BOOST_AUTO_TEST_CASE( grid_1d_read ) {
  DimmedGrid<1> g(GRID_SRC + "/1.grid");
  BOOST_REQUIRE_EQUAL(g.min_[0], 0);
  BOOST_REQUIRE_EQUAL(g.max_[0], 2.5 + g.dx_[0]);
  BOOST_REQUIRE_EQUAL(g.grid_number_[0], 101);
  BOOST_REQUIRE_EQUAL(g.grid_number_[0], 101);
}

BOOST_AUTO_TEST_CASE( grid_3d_read ) {
  DimmedGrid<3> g(GRID_SRC + "/3.grid");
  BOOST_REQUIRE_EQUAL(g.min_[2], 0);
  BOOST_REQUIRE_EQUAL(g.max_[2], 2.5 + g.dx_[2]);
  BOOST_REQUIRE_EQUAL(g.grid_number_[2], 11);
  double temp[] = {0.75, 0, 1.00};
  BOOST_REQUIRE(pow(g.get_value(temp) - 1.260095, 2) < EPSILON);
  
}

BOOST_AUTO_TEST_CASE( derivative_direction ) {
  DimmedGrid<3> g(GRID_SRC + "/3.grid");
  g.b_interpolate_ = 1;

  double temp[] = {0.75, 0, 1.00};
  double temp2[] = {0.76, 0, 1.00};
  BOOST_REQUIRE(g.get_value(temp2)> g.get_value(temp));
  temp2[0] = 0.75;
  temp2[2] = 0.99;
  BOOST_REQUIRE(g.get_value(temp2) < g.get_value(temp));
  
}



BOOST_AUTO_TEST_CASE( grid_read_write_consistency ) {

  size_t i, j;
  std::string input;
  std::string output;
  for(i = 1; i <= 3; i++) {
    std::stringstream filename;
    filename << i << ".grid";
    input = GRID_SRC + "/" + filename.str();
    output = filename.str() + ".test";
    Grid* g;
    switch(i) {
    case 1:
      g = new DimmedGrid<1>(input);
      break;
    case 2:
      g = new DimmedGrid<2>(input);
      break;
    case 3:
      g = new DimmedGrid<3>(input);
      break;
    }
    g->write(output);
    //grab the grid for comparison
    size_t ref_length = g->get_grid_size();
    double ref_grid[ref_length];
    for(j = 0; j < ref_length; j++)
      ref_grid[j] = g->get_grid()[j];
    //re-read
    g->read(output);
    //now compare
    BOOST_REQUIRE_EQUAL(g->get_grid_size(), ref_length);

    for(j = 0; j < ref_length; j++)
      BOOST_REQUIRE(pow(ref_grid[j] - g->get_grid()[j], 2) < EPSILON);

  }
  
}

BOOST_AUTO_TEST_CASE( interpolation_1d ) {
  
  double min[] = {0};
  double max[] = {10};
  double bin_spacing[] = {1};
  int periodic[] = {0};
  DimmedGrid<1> g (min, max, bin_spacing, periodic, 1, 1);
  //std::cout << "I'm in interpolation_1d test and g.grid_number_[0] is now " <<g.grid_number_[0] << ".\n";//this comes out to be 11 for both max_[0] and grid_number_[0]
  
  for(int i = 0; i < 11; i++) {
    g.grid_[i] = log(i);
    g.grid_deriv_[i] = 1. / i;
  }

  double array[] = {5.3};
  double der[1];
  double fhat = g.get_value_deriv(array,der);

  //make sure it's at least in the ballpark
  BOOST_REQUIRE(fhat > log(5) && fhat < log(6));
  BOOST_REQUIRE(der[0] < 1. / 5 && der[0] > 1. / 6.);

  //Make sure it's reasonably accurate
  BOOST_REQUIRE(pow(fhat - log(5.3), 2) < 0.1);
  BOOST_REQUIRE(pow(der[0]- 1. / 5.3, 2) < 0.1);

  //try edge cases
  array[0] = 5.0;
  g.get_value(array);
  array[0] = 5.5;
  g.get_value(array);
  array[0] = 0.0;
  g.get_value(array);
  array[0] = 10.0;
  g.get_value(array);//[rainier] problem with getting array value at rightmost edge, not leftmost

}

BOOST_AUTO_TEST_CASE( interp_1d_periodic ) {
  double min[] = {-M_PI};
  double max[] = {M_PI};
  double bin_spacing[] = {M_PI / 100};
  int periodic[] = {1};
  DimmedGrid<1> g (min, max, bin_spacing, periodic, 1, 1);

  for(int i = 0; i < g.grid_number_[0]; i++) {
    g.grid_[i] = sin(g.min_[0] + i * g.dx_[0]);
    g.grid_deriv_[i] = cos(g.min_[0] + i * g.dx_[0]);
  }


  double array[] = {M_PI / 4};
  double der[1];
  double fhat = g.get_value_deriv(array,der);

  //Make sure it's reasonably accurate
  BOOST_REQUIRE(pow(fhat - sin(array[0]), 2) < 0.1);
  BOOST_REQUIRE(pow(der[0] - cos(array[0]), 2) < 0.1);

  //test periodic
  array[0] = 5 * M_PI / 4;
  fhat = g.get_value_deriv(array,der);

  g.write("grid.test");
  
  BOOST_REQUIRE(pow(fhat - sin(array[0]), 2) < 0.1);
  BOOST_REQUIRE(pow(der[0] - cos(array[0]), 2) < 0.1);

}

BOOST_AUTO_TEST_CASE( boundary_remap_wrap) {

  //this test simulates a subdivision that is periodic and stretches across the box in 1D
  //and is non-periodic and partial in the other

  double min[] = {0, 0};
  double max[] = {10, 5};
  double bin_spacing[] = {1, 1};
  int periodic[] = {1, 0, 0};
  double sigma[] = {0.1, 0.1};
  DimmedGaussGrid<2> g (min, max, bin_spacing, periodic, 1, sigma);
  max[1] = 10;
  periodic[1] = 1;
  g.set_boundary(min, max, periodic);

  double test_point[] = {0,1}; //should not remap
  g.remap(test_point);
  BOOST_REQUIRE(pow(test_point[0] - 0, 2) < 0.1);
  BOOST_REQUIRE(pow(test_point[1] - 1, 2) < 0.1);

  test_point[0] = -1;//on grid, at 9
  g.remap(test_point);
  BOOST_REQUIRE(pow(test_point[0] - 9, 2) < 0.1);
  BOOST_REQUIRE(pow(test_point[1] - 1, 2) < 0.1);

  test_point[1] = 6;//closest point is 6
  g.remap(test_point);
  BOOST_REQUIRE(pow(test_point[0] - 9, 2) < 0.1);
  BOOST_REQUIRE(pow(test_point[1] - 6, 2) < 0.1);

  test_point[1] = 11;//actually in grid at 1
  g.remap(test_point);
  BOOST_REQUIRE(pow(test_point[0] - 9, 2) < 0.1);
  BOOST_REQUIRE(pow(test_point[1] - 1, 2) < 0.1);

  test_point[1] = 9; //closest point is -1
  g.remap(test_point);
  BOOST_REQUIRE(pow(test_point[0] - 9, 2) < 0.1);
  BOOST_REQUIRE(pow(test_point[1] - -1, 2) < 0.1);

  test_point[1] = -1; //closest point is -1
  g.remap(test_point);
  BOOST_REQUIRE(pow(test_point[0] - 9, 2) < 0.1);
  BOOST_REQUIRE(pow(test_point[1] - -1, 2) < 0.1);


}

BOOST_AUTO_TEST_CASE( boundary_remap_wrap_2) {

  //this test simulates a subdivision that is periodic and stretches across the box in 1D
  //and is non-periodic and partial in the other

  double min[] = {-2};
  double max[] = {7};
  double bin_spacing[] = {0.1};
  int periodic[] = {0};
  double sigma[] = {0.1};
  DimmedGaussGrid<1> g (min, max, bin_spacing, periodic, 1, sigma);
  min[0] = 0;
  max[0] = 10;
  periodic[0] = 1;
  g.set_boundary(min, max, periodic);

  double test_point[] = {0}; //should not remap
  g.remap(test_point);
  BOOST_REQUIRE(pow(test_point[0] - 0, 2) < 0.1);

  test_point[0] = -1;//shoul not remap
  g.remap(test_point);
  BOOST_REQUIRE(pow(test_point[0] - -1, 2) < 0.1);

  test_point[0] = 9;//should remap
  g.remap(test_point);
  BOOST_REQUIRE(pow(test_point[0] - -1, 2) < 0.1);

  test_point[0] = 6;//should not remap
  g.remap(test_point);
  BOOST_REQUIRE(pow(test_point[0] - 6, 2) < 0.1);


}


BOOST_AUTO_TEST_CASE( boundary_remap_wrap_3) {

  //this test simulates a subdivision that is periodic and stretches across the box in 1D
  //and is non-periodic and partial in the other

  double min[] = {-2};
  double max[] = {7};
  double bin_spacing[] = {0.1};
  int periodic[] = {0};
  double sigma[] = {0.1};
  DimmedGaussGrid<1> g (min, max, bin_spacing, periodic, 1, sigma);
  min[0] = 0;
  max[0] = 10;
  periodic[0] = 1;
  g.set_boundary(min, max, periodic);

  double point[] = {0.01};
  g.add_value(point,1);
  double der[1];
  point[0] = 0;
  g.get_value_deriv(point, der);
  BOOST_REQUIRE(fabs(der[0]) > 0.1);


}


BOOST_AUTO_TEST_CASE( boundary_remap_nowrap_1) {

  //this test simulates a subdivision that is periodic and stretches across the box in 1D
  //and is non-periodic and partial in the other

  double min[] = {-2};
  double max[] = {7};
  double bin_spacing[] = {0.1};
  int periodic[] = {0};
  double sigma[] = {0.1};
  DimmedGaussGrid<1> g (min, max, bin_spacing, periodic, 1, sigma);
  min[0] = 0;
  max[0] = 10;
  periodic[0] = 0;
  g.set_boundary(min, max, periodic);

  double point[] = {-0.01};
  g.add_value(point,1);
  double der[1];
  point[0] = 0;
  g.get_value_deriv(point, der);
  BOOST_REQUIRE(fabs(point[0]) < EPSILON);


}




BOOST_AUTO_TEST_CASE( interp_3d_mixed ) {
  double min[] = {-M_PI, -M_PI, 0};
  double max[] = {M_PI, M_PI, 10};
  double bin_spacing[] = {M_PI / 100, M_PI / 100, 1};
  int periodic[] = {1, 1, 0};
  DimmedGrid<3> g (min, max, bin_spacing, periodic, 1, 0);
  
  size_t index = 0;
  double x,y,z;
  
  for(int i = 0; i < g.grid_number_[2]; i++) {
    for(int j = 0; j < g.grid_number_[1]; j++) {
      for(int k = 0; k < g.grid_number_[0]; k++) {
	x = g.min_[0] + k * g.dx_[0];
	y = g.min_[1] + j * g.dx_[1];
	z = g.min_[2] + i * g.dx_[2];
	g.grid_[index] = cos(x) * sin(y) * z;
	g.grid_deriv_[index * 3 + 0] = -sin(x) * sin(y) * z;
	g.grid_deriv_[index * 3 + 1] = cos(x) * cos(y) * z;
	g.grid_deriv_[index * 3 + 2] = cos(x) * sin(y);
	index++;
      }
    }
  }

  double array[] = {-10.75 * M_PI / 2, 8.43 * M_PI / 2, 3.5};
  double der[3];
  double fhat = g.get_value_deriv(array,der);
  double f = cos(array[0]) * sin(array[1]) * array[2];
  double true_der[] = {-sin(array[0]) * sin(array[1]) * array[2],
		       cos(array[0]) * cos(array[1]) * array[2],
		       cos(array[0]) * sin(array[1])};
  
  BOOST_REQUIRE(pow(f- fhat, 2) < 0.1);
  BOOST_REQUIRE(pow(der[0] - true_der[0], 2) < 0.1);
  BOOST_REQUIRE(pow(der[1] - true_der[1], 2) < 0.1);
  BOOST_REQUIRE(pow(der[2] - true_der[2], 2) < 0.1);

}

BOOST_AUTO_TEST_CASE( gauss_grid_add_check ) {
  double min[] = {-10};
  double max[] = {10};
  double sigma[] = {1};
  double bin_spacing[] = {1};
  int periodic[] = {1};
  DimmedGaussGrid<1> g (min, max, bin_spacing, periodic, 0, sigma);

  //add 1 gaussian
  double x[] = {0};
  g.add_value(x, 1);

  //now check a few points
  BOOST_REQUIRE(pow(g.get_value(x) - 1 / sqrt(2 * M_PI), 2) < EPSILON);
  
  int i;
  double der[1];
  double value;
  for( i = -6; i < 7; i++) {
    x[0] = i;
    value = g.get_value_deriv(x, der);
    BOOST_REQUIRE(pow(value - exp(-x[0]*x[0]/2.) / sqrt(2*M_PI), 2) < 0.01);
    BOOST_REQUIRE(pow(der[0] - (-x[0] *exp(-x[0]*x[0]/2.)) / sqrt(2*M_PI), 2) < 0.01);
  }
 
}


BOOST_AUTO_TEST_CASE( gauss_pbc_check ) {
  double min[] = {2};
  double max[] = {10};
  double sigma[] = {1};
  double bin_spacing[] = {1};
  int periodic[] = {1};
  DimmedGaussGrid<1> g (min, max, bin_spacing, periodic, 0, sigma);

  //add 1 gaussian
  double x[] = {2};
  g.add_value(x, 1);

  int i;
  double der[1];
  double value;
  double dx;
  for( i = -6; i < 7; i++) {
    x[0] = i;
    dx = x[0] - 2;
    dx  -= round(dx / (min[0] - max[0])) * (min[0] - max[0]);
    value = g.get_value_deriv(x, der);

    std::cout << "x = " << x[0]
	      << " dx = " << dx 
	      << "(" 
	      << " value = " << value 
	      << " (" << exp(-dx*dx/2.) / sqrt(2 * M_PI) << ")" 
	      << std::endl;

    BOOST_REQUIRE(pow(value - exp(-dx*dx/2.) / sqrt(2 * M_PI), 2) < 0.01);
    BOOST_REQUIRE(pow(der[0] - (-dx *exp(-dx*dx/2.)) / sqrt(2 * M_PI), 2) < 0.01);
  }
 
}


BOOST_AUTO_TEST_CASE( gauss_subdivided_pbc_check ) {
  double min[] = {2};
  double max[] = {4};
  double sigma[] = {1};
  double bin_spacing[] = {1};
  int periodic[] = {0};
  double gauss_loc[] = {11};
  double x[1];
  DimmedGaussGrid<1> g (min, max, bin_spacing, periodic, 0, sigma);
  periodic[0] = 1;
  max[0] = 10;
  g.set_boundary(min, max, periodic);

  //add 1 gaussian
  g.add_value(gauss_loc, 1); //added at equivalent to 1

  int i;
  double der[1];
  double value;
  double dx;
  for( i = 2; i < 4; i++) {
    x[0] = i;
    dx = x[0] - gauss_loc[0];
    dx  -= round(dx / (min[0] - max[0])) * (min[0] - max[0]);
    value = g.get_value_deriv(x, der);

    /*
    std::cout << "x = " << x[0]
	      << " dx = " << dx 
	      << " value = " << value 
	      << " (" << exp(-dx*dx/2.) << ")" 
	      << std::endl;
    */

    BOOST_REQUIRE(pow(value - exp(-dx*dx/2.) / sqrt(2 * M_PI), 2) < 0.01);
    BOOST_REQUIRE(pow(der[0] - (-dx *exp(-dx*dx/2.)) / sqrt(2 * M_PI), 2) < 0.01);
  }
 
}


BOOST_AUTO_TEST_CASE( gauss_grid_integral_test ) {
  double min[] = {-100};
  double max[] = {100};
  double sigma[] = {1.2};
  double bin_spacing[] = {1};
  int periodic[] = {1};
  DimmedGaussGrid<1> g (min, max, bin_spacing, periodic, 1, sigma);

  //add N gaussian
  int N = 20;
  int i;
  double x[1];
  double offsets = 1. / N;
  double g_integral = 0;

  //generate a random number but use sequential grid point offsets
  for(i = 0; i < N; i++) {
    x[0] = rand() % 200 - 100 + i * offsets;
    g_integral += g.add_value(x, 1.5);
  }

  //now we integrate the grid
  double area = 0;
  double dx = 0.1;
  int bins = (int) 200 / dx;
  for(i = 0; i < bins; i++) {
    x[0] = -100 + i * dx;
    area += g.get_value(x) * dx;
  }

  //Make sure the integrated area is correct
  //unnormalized, so a little height scaling is necessary
  //  std::cout << area << " " << N * 1.5 << std::endl;
  BOOST_REQUIRE(pow(area - N * 1.5, 2) < 1);

  //now make sure that add_value returned the correct answers as well
   BOOST_REQUIRE(pow(area - g_integral, 2) < 0.1);
}

BOOST_AUTO_TEST_CASE( gauss_grid_integral_test_mcgdp ) {
  double min[] = {-100};
  double max[] = {100};
  double sigma[] = {10};
  double bin_spacing[] = {1};
  int periodic[] = {0};
  DimmedGaussGrid<1> g (min, max, bin_spacing, periodic, 1, sigma);

  //add N gaussian
  int N = 20;
  int i;
  double x[1];
  double offsets = 1. / N;
  double g_integral = 0;
  double temp;

  //get boundaries
  x[0] = -100.0;
  temp = g.add_value(x, 1.5);
  std::cout << x[0] << ": " << temp  << " == " << 1.5 << std::endl;
  g_integral += temp;		      

  x[0] = 100.0;
  temp = g.add_value(x, 1.5);
  std::cout << x[0] << ": " << temp  << " == " << 1.5 << std::endl;
  g_integral += temp;		      


  //generate a random number but use sequential grid point offsets
  for(i = 0; i < N; i++) {
    x[0] = rand() % 200 - 100 + i * offsets;
    temp = g.add_value(x, 1.5);
    std::cout << x[0] << ": " << temp  << " == " << 1.5 << std::endl;
    g_integral += temp;		      
  }

  //now we integrate the grid
  double area = 0;
  double dx = 0.1;
  int bins = (int) 200 / dx;
  for(i = 0; i < bins; i++) {
    x[0] = -100 + i * dx;
    area += g.get_value(x) * dx;
  }

  //Make sure the integrated area is correct
  //unnormalized, so a little height scaling is necessary
  std::cout << area / N << " " << 1.5  << std::endl;
  BOOST_REQUIRE(pow(area - N * 1.5, 2) < 1);

  //now make sure that add_value returned the correct answers as well
   BOOST_REQUIRE(pow(area - g_integral, 2) < 0.1);
}


BOOST_AUTO_TEST_CASE( gauss_grid_derivative_test ) {
  double min[] = {-100};
  double max[] = {100};
  double sigma[] = {1.2};
  double bin_spacing[] = {1};
  int periodic[] = {1};
  DimmedGaussGrid<1> g (min, max, bin_spacing, periodic, 1, sigma);

  //add N gaussian
  int N = 20;
  int i;
  double x[1];
  double offsets = 1. / N;
  double g_integral = 0;

  //generate a random number but use sequential grid point offsets
  for(i = 0; i < N; i++) {
    x[0] = rand() % 200 - 100 + i * offsets;
    g_integral += g.add_value(x, 1.5);
  }

  //now we calculate finite differences on the grid
  double vlast, vlastlast, v, approx_der;  

  double der[1];
  double der_last;
  double dx = 0.1;
  int bins = (int) 200 / dx;
  for(i = 0; i < bins; i++) {
    x[0] = -100 + i * dx;
    v = g.get_value_deriv(x,der);
    if(i > 1) {
      approx_der = (v - vlastlast) / (2*dx);
      BOOST_REQUIRE(pow(approx_der - der_last, 2) < 0.01);
    }
    vlastlast = vlast;
    vlast = v;

    der_last = der[0];
  }

}

BOOST_AUTO_TEST_CASE( gauss_grid_derivative_test_mcgdp_1 ) {
  double min[] = {-100};
  double max[] = {100};
  double sigma[] = {1.2};
  double bin_spacing[] = {1};
  int periodic[] = {0};
  DimmedGaussGrid<1> g (min, max, bin_spacing, periodic, 1, sigma);

  //add N gaussian
  int N = 20;
  int i;
  double x[1];
  double offsets = 1. / N;
  double g_integral = 0;

  //generate a random number but use sequential grid point offsets
  for(i = 0; i < N; i++) {
    x[0] = rand() % 200 - 100 + i * offsets;
    g_integral += g.add_value(x, 1.5);
  }

  //now we calculate finite differences on the grid
  double vlast, vlastlast, v, approx_der;  

  double der[1];
  double der_last;
  double dx = 0.1;
  int bins = (int) 200 / dx;
  for(i = 0; i < bins; i++) {
    x[0] = -100 + i * dx;
    v = g.get_value_deriv(x,der);
    if(i > 1) {
      approx_der = (v - vlastlast) / (2*dx);
      BOOST_REQUIRE(pow(approx_der - der_last, 2) < 0.001);
    } else {
      BOOST_REQUIRE(pow(der[0], 2) < 0.001);
    }
    vlastlast = vlast;
    vlast = v;

    der_last = der[0];
  }
  
  approx_der = (vlast - vlastlast) / dx;
  BOOST_REQUIRE(pow(approx_der - der_last, 2) < 0.1);
  BOOST_REQUIRE(pow(der_last, 2) < 0.01);

}

BOOST_AUTO_TEST_CASE( gauss_grid_interp_test_mcgdp_1D ) {
  double min[] = {-100};
  double max[] = {100};
  double sigma[] = {10.0};
  double bin_spacing[] = {1};
  int periodic[] = {1};
  DimmedGaussGrid<1> g (min, max, bin_spacing, periodic, 1, sigma);
  periodic[0]  = 0;
  min[0] = -50;
  max[0] = 50;
  g.set_boundary(min, max, periodic);

  //add N gaussian
  int N = 20;
  int i;
  double x[1];
  double der[1];
  double v;

  //generate a random number
  for(i = 0; i < N; i++) {
    x[0] = rand() % 200 - 100;
    g.add_value(x, 1.0);
  }

  //Check if the boundaries were duplicated
  BOOST_REQUIRE(pow(g.grid_.grid_[50] - g.grid_.grid_[49] ,2) < EPSILON);
  BOOST_REQUIRE(pow(g.grid_.grid_[150] - g.grid_.grid_[151] ,2) < EPSILON);

  x[0] = 50.1;
  v = g.get_value(x);
  x[0] = 50.0;
  BOOST_REQUIRE(pow(v - g.get_value(x),2) < EPSILON);

  //boundaries should be 0, even with interpolation
  g.get_value_deriv(x,der);  
  BOOST_REQUIRE(der[0] * der[0] < EPSILON);

  //check other side
  x[0] = -50.1;
  v = g.get_value(x);
  x[0] = -50.0;
  BOOST_REQUIRE(pow(v - g.get_value(x),2) < EPSILON);
  g.get_value_deriv(x,der);  
  BOOST_REQUIRE(der[0] * der[0] < EPSILON);

}

BOOST_AUTO_TEST_CASE( gauss_grid_interp_test_mcgdp_3D ) {
  double min[] = {-10, -10, -10};
  double max[] = {10, 10, 10};
  double sigma[] = {3.0, 3.0, 3.0};
  double bin_spacing[] = {0.9, 1.1, 1.4};
  int periodic[] = {1, 1, 1};
  DimmedGaussGrid<3> g (min, max, bin_spacing, periodic, 1, sigma);
  periodic[0]  = periodic[1] = periodic[2] = 0;
  min[0]  = min[1] = min[2] = -5;
  max[0]  = max[1] = max[2] = 5;
  g.set_boundary(min, max, periodic);

  //add N gaussian
  int N = 20;
  int i;
  double x[3];
  double der[3];
  double v;

  //generate a random number
  for(i = 0; i < N; i++) {
    x[0] = rand() % 20 - 10;
    x[1] = rand() % 20 - 10;
    x[2] = rand() % 20 - 10;
    g.add_value(x, 5.0);
  }

  //Check if the boundaries were duplicated
  x[0] = x[2] = 50.1;
  x[1] = 5.0;
  v = g.get_value(x);
  x[0] = x[1] = 50.0;
  BOOST_REQUIRE(pow(v - g.get_value(x),2) < EPSILON);

  //boundaries should be 0, even with interpolation
  g.get_value_deriv(x,der);  
  BOOST_REQUIRE(der[0] * der[0] < 0.001);

  //check another location
  x[0] = -5.1;
  x[2] = 5.1;
  v = g.get_value(x);
  x[0] = x[2] = -5.0;
  BOOST_REQUIRE(pow(v - g.get_value(x),2) < 0.001);
  g.get_value_deriv(x,der);  
  BOOST_REQUIRE(der[0] * der[0] < EPSILON);

}




BOOST_AUTO_TEST_CASE( gauss_grid_integral_regression_1 ) {
  double min[] = {0};
  double max[] = {10};
  double bin_spacing[] = {0.009765625};
  double sigma[] = {0.1};
  int periodic[] = {1};
  GaussGrid* g  = make_gauss_grid(1, min, max, bin_spacing, periodic, 1, sigma);
  periodic[0] = 1;
  g->set_boundary(min, max, periodic);

  //add gaussian that was failing
  double x[] = {-3.91944};
  double h = 1.0;
  double bias_added = g->add_value(x, h);

  //unnormalized, so a little height scaling is necessary
  //std::cout << bias_added /  (sqrt(2 * M_PI) * sigma[0]) << " " << h << std::endl;
  BOOST_REQUIRE(pow(bias_added - h, 2) < 0.1);

  delete g;
}


BOOST_AUTO_TEST_CASE( edm_bias_reader ) {
  EDMBias bias(EDM_SRC + "/read_test.edm");
  BOOST_REQUIRE_EQUAL(bias.dim_, 2);
  BOOST_REQUIRE_EQUAL(bias.b_tempering_, 0);
  BOOST_REQUIRE(pow(bias.bias_sigma_[0] - 2,2) < EPSILON);
  BOOST_REQUIRE(pow(bias.bias_dx_[1] - 1.0,2) < EPSILON);
}



struct EDMBiasTest {
  EDMBiasTest() : bias(EDM_SRC + "/sanity.edm") {
        
    bias.setup(1, 1);
    double low[] = {0};
    double high[] = {10};
    int p[] = {1};
    double skin[] = {0};
    bias.subdivide(low, high, low, high, p, skin);
    
  }     
  
  EDMBias bias;
};

BOOST_FIXTURE_TEST_SUITE( edmbias_test, EDMBiasTest )

BOOST_AUTO_TEST_CASE( edm_sanity ) {
  double** positions = (double**) malloc(sizeof(double*));
  positions[0] = (double*) malloc(sizeof(double));
  double runiform[] = {1};
  
  positions[0][0] = 5.0;
  bias.add_hills(1, positions, runiform);

  bias.write_bias("BIAS");

  
  //  std::cout << bias.hill_prefactor_ / sqrt(2 * M_PI) / bias.bias_sigma_[0] << " " 
  //	    << bias.bias_->get_value(positions[0]) << std::endl;
  //test if the value at the point is correct
  BOOST_REQUIRE(pow(bias.bias_->get_value(positions[0]) - bias.hill_prefactor_ / sqrt(2 * M_PI) / bias.bias_sigma_[0], 2) < EPSILON);
  //check if the claimed amount of bias added is correct
  BOOST_REQUIRE(pow(bias.cum_bias_ - bias.hill_prefactor_, 2) < 0.001);
  
  //now  check that the forces point away from the hills
  double der[0];
  positions[0][0] = 4.99; //to the left
  bias.bias_->get_value_deriv(positions[0], der);
  //the negative of the bias (the force) should point to the left
  BOOST_REQUIRE(-der[0] < 0);

  positions[0][0] = 5.01; // to the right of the bias
  bias.bias_->get_value_deriv(positions[0], der);
  //the negative of the bias (the force) should point to the right
  BOOST_REQUIRE(-der[0] > 0);

  

  free(positions[0]);
  free(positions);
}

double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
  }

//This test will simply run several thousand timesteps and time how long it takes.
BOOST_AUTO_TEST_CASE(edm_cpu_timer_1d){
  
  double iStart, iElapsed;
  double min[] = {-10};
  double max[] = {10};
  double sigma[] = {1};
  double bin_spacing[] = {1};
  int periodic[] = {1};
  double x[1] = {0};
  unsigned int n_hills = 5000000;
  DimmedGaussGrid<1> g (min, max, bin_spacing, periodic, 0, sigma);
  //now just do a generic loop, adding 10k gaussians, and time it
  iStart = cpuSecond();
  for( unsigned int i = 0; i < n_hills; i++){
    int rand_num = rand() % 20 - 10;
    x[0] = rand_num;
    g.add_value(x,1);
  }
  iElapsed = cpuSecond() - iStart;
  printf("Time elapsed for adding %u hills: %f sec\n", n_hills, iElapsed);
}


BOOST_AUTO_TEST_SUITE_END()
