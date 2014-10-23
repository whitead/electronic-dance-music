#include "grid.h"
#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE EDM
#include <boost/test/unit_test.hpp>
#include <boost/timer.hpp>
#include <sstream>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>


#define EPSILON 1e-10
#define QUOTE(name) #name
#define STR(macro) QUOTE(macro)
#define GRID_SRC std::string(STR(TEST_GRID_SRC))

using namespace boost;

BOOST_AUTO_TEST_CASE( grid_1d_sanity )
{
  /* Visual:
   * valuess:  0   1   2   3
   * grid:   |---|---|---|---|---
   *         0   1   2   3   4
   */
  double min[] = {0};
  double max[] = {10};
  double bin_spacing[] = {1};
  int periodic[] = {0};
  DimmedGrid<1> g (min, max, bin_spacing, periodic, 0, 0);
  g.initialize();

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
  g.initialize();

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
  double temp[] = {0.76, 0, 1.01};
  BOOST_REQUIRE(pow(g.get_value(temp) - 1.260095, 2) < EPSILON);
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
    double *ref_grid = g->grid_;
    g->grid_ = NULL;
    size_t ref_length = g->grid_size_;
    //re-read
    g->read(output);
    //now compare
    BOOST_REQUIRE_EQUAL(g->grid_size_, ref_length);

    for(j = 0; j < ref_length; j++)
      BOOST_REQUIRE(pow(ref_grid[i] - g->grid_[i], 2) < EPSILON);

    free(ref_grid);    
  }
  
}

BOOST_AUTO_TEST_CASE( interpolation_1d ) {
  
  double min[] = {0};
  double max[] = {10};
  double bin_spacing[] = {1};
  int periodic[] = {0};
  DimmedGrid<1> g (min, max, bin_spacing, periodic, 1, 1);
  g.initialize();

  
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
  g.get_value(array);

}

BOOST_AUTO_TEST_CASE( interp_1d_periodic ) {
  double min[] = {-M_PI};
  double max[] = {M_PI};
  double bin_spacing[] = {M_PI / 100};
  int periodic[] = {1};
  DimmedGrid<1> g (min, max, bin_spacing, periodic, 1, 1);
  g.initialize();

  for(int i = 0; i < g.grid_number_[0]; i++) {
    g.grid_[i] = sin(g.min_[0] + i * g.dx_[0]);
    g.grid_deriv_[i] = cos(g.min_[0] + i * g.dx_[0]);
  }

  double array[] = {M_PI / 4};
  double der[1];
  double fhat = g.get_value_deriv(array,der);

  //Make sure it's reasonably accurate
  BOOST_REQUIRE(pow(fhat- sin(array[0]), 2) < 0.1);
  BOOST_REQUIRE(pow(der[0] - cos(array[0]), 2) < 0.1);

  //test periodic
  array[0] = 5 * M_PI / 4;
  fhat = g.get_value_deriv(array,der);

  BOOST_REQUIRE(pow(fhat - sin(array[0]), 2) < 0.1);
  BOOST_REQUIRE(pow(der[0] - cos(array[0]), 2) < 0.1);

}

BOOST_AUTO_TEST_CASE( interp_3d_mixed ) {
  double min[] = {-M_PI, -M_PI, 0};
  double max[] = {M_PI, M_PI, 10};
  double bin_spacing[] = {M_PI / 100, M_PI / 100, 1};
  int periodic[] = {1, 1, 0};
  DimmedGrid<3> g (min, max, bin_spacing, periodic, 1, 0);
  g.initialize();
  
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

struct FooTest {
  FooTest(){
    //
  }  
};

BOOST_FIXTURE_TEST_SUITE( foo_test, FooTest )


BOOST_AUTO_TEST_SUITE_END()
