#ifndef EDM_BIAS_PY_H_
#define EDM_BIAS_PY_H_


#include "edm_bias.h"
#include <boost/python.hpp>

namespace bpy = boost::python;

namespace EDM{
  class EDMBias_Py : public EDMBias {
    
  public:
    EDMBias_Py(const std::string& input_filename, double temperature, double boltzmann_constant);
    EDMBias_Py(const EDMBias_Py& that);//disable copy constructor
    
    void subdivide_py(bpy::list boxlo, bpy::list boxhi, bpy::list periodic);

    void add_hill_py(bpy::list position, double runiform);

    bpy::tuple get_force_py(bpy::list position);
  private:
    double* pos_buffer_;
    double* force_buffer_;

  };

}
#endif// EDM_BIAS_PY_H_
