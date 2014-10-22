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

#define EPSILON 1e-5

using namespace boost;

struct FooTest {
  FooTest(){
    //
  }  
};

BOOST_FIXTURE_TEST_SUITE( foo_test, FooTest )

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
  Grid<1> g (min, max, bin_spacing, periodic, 0);
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
    
}

BOOST_AUTO_TEST_CASE( grid_3d_sanity )
{
  double min[] = {-2, -5, -3};
  double max[] = {125, 63, 78};
  double bin_spacing[] = {1.27, 1.36, 0.643};
  int periodic[] = {0, 1, 1};
  Grid<3> g (min, max, bin_spacing, periodic, 0);
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
  Grid<1>("sample_1d.grid");
  BOOST_REQUIRE_EQUAL(g.min_[0] == 0)
  BOOST_REQUIRE_EQUAL(g.max_[0] == 2.5 + g.dx_[0])
  BOOST_REQUIRE_EQUAL(g.grid_number_[0] == 101)
  BOOST_REQUIRE_EQUAL(g.grid_number_[0] == 101)
}


BOOST_AUTO_TEST_SUITE_END()
