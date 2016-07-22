#include "edm_bias_py.h"

//conversion code
namespace bpy = boost::python;

void convert_list(bpy::list o, double* array, unsigned int length) {
  std::size_t n = bpy::len(o);
  if(n > length)
    EDM::edm_error("Tried to convert a list that was too big", "edm_bias_py.cpp:convert_list");  
  for (int i = 0; i < n; i++) {
    array[i] = bpy::extract<double>(o[i]);
  }
}



//call super constructor 
EDM::EDMBias_Py::EDMBias_Py(const std::string& input_filename,
			    double temperature,
			    double boltzmann_constant): EDMBias(input_filename) {

  pos_buffer_ = (double*) malloc(sizeof(double) * dim_);
  force_buffer_ = (double*) malloc(sizeof(double) * dim_);
  setup(temperature, boltzmann_constant);
};


//simplified subdivide command
void EDM::EDMBias_Py::subdivide_py(bpy::list boxlo, bpy::list boxhi, bpy::list periodic) {
  std::size_t n = bpy::len(boxlo);

  //change this over to n-dimensions once I change the subdivide command as well.
  double c_boxlo[3];
  double c_boxhi[3];
  int b_periodic[3];
  double skin[3];
  int i;
  
  for (i = 0; i < n; i++) {
    c_boxlo[i] = bpy::extract<double>(boxlo[i]);
    c_boxhi[i] = bpy::extract<double>(boxhi[i]);
    b_periodic[3] = bpy::extract<int>(periodic[i]);
    skin[i]  = 0.0;
  }

  subdivide(c_boxlo, c_boxhi, c_boxlo, c_boxhi, b_periodic,skin);
  
}


void EDM::EDMBias_Py::add_hill_py(bpy::list position, double runiform) {
  int i;
  
  for (i = 0; i < dim_; i++) {
    pos_buffer_[i] = bpy::extract<double>(position[i]);
  }

  add_hill(pos_buffer_, runiform);
  
}


bpy::tuple EDM::EDMBias_Py::get_force_py(bpy::list position) {

  int i;
  
  for (i = 0; i < dim_; i++) {    
    pos_buffer_[i] = bpy::extract<double>(position[i]);
    force_buffer_[i] = 0; //also zero out force buffer
  }

  double e = bias_->get_value_deriv(pos_buffer_, force_buffer_);
  bpy::list force;
  for(i = 0; i < dim_; i++)
    force.append(force_buffer_[i]);

  return bpy::make_tuple(e, force);
  
}
