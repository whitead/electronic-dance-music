#include <boost/python.hpp>
#include "edm_bias_py.h"
using namespace boost::python;
using namespace EDM;

BOOST_PYTHON_MODULE(libedm_python)
{
  class_<EDMBias_Py, boost::noncopyable>("EDMBias_Py", init<std::string, double, double>())
    .def("set_box", &EDMBias_Py::subdivide_py)
    .def("pre_add_hill", &EDMBias_Py::pre_add_hill)
    .def("post_add_hill", &EDMBias_Py::post_add_hill)
    .def("add_hill_r", &EDMBias_Py::add_hill_py)
    .def("write_bias", &EDMBias_Py::write_bias)
    .def("write_lammps_table", &EDMBias_Py::write_lammps_table)
    .def("write_histogram", &EDMBias_Py::write_histogram)
    .def("clear_histogram", &EDMBias_Py::clear_histogram)
    .def("get_force", &EDMBias_Py::get_force_py)
    ;
}
