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

  for(int i = 0; i < 10; i++)
    g.grid_[i] = i;

  double x[] = {3.5};
  BOOST_REQUIRE(pow(g.get_value(x) - 3, 2) < 0.000001);
    
}

BOOST_AUTO_TEST_SUITE_END()
