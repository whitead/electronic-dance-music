#include "grid.h"
#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE WordModel
#include <boost/test/unit_test.hpp>
#include <boost/timer.hpp>
#include <sstream>
#include <iostream>
#include <fstream>
#include <ctime>

using namespace wordmodel;
using namespace std;
using namespace boost;

struct FooTest {
  FooTests(){
    //
  }  
};

BOOST_FIXTURE_TEST_SUITE( foo_test, FooTest )

BOOST_AUTO_TEST_CASE( foo_test_1 )
{

  BOOST_REQUIRE( 2 == 2 );
}

BOOST_AUTO_TEST_SUITE_END()
