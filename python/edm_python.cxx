#include <boost/python.hpp>
#include "edm_bias.h"
using namespace boost::python;
using namespace EDM;

BOOST_PYTHON_MODULE(libedm_python)
{
  class_<EDMBias, boost::noncopyable>("EDMBias", init<std::string>())
      .def("subdivide", &EDMBias::subdivide)
      .def("pre_add_hill", &EDMBias::pre_add_hill)
      .def("post_add_hill", &EDMBias::post_add_hill)
      .def("add_hill", &EDMBias::add_hill)
      .def("write_bias", &EDMBias::write_bias)
      .def("write_lammps_table", &EDMBias::write_lammps_table)
      .def("write_histogram", &EDMBias::write_histogram)
      .def("clear_histogram", &EDMBias::clear_histogram)
      .def("update_force", &EDMBias::update_force)
    ;
}
