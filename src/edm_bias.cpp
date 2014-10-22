#include "edm_bias.h"

EDMBias::EDMBias(std::string& input_filename) {
  
  //set defaults
  b_tempering = 0;
  global_tempering = 0;
  hill_density = -1;
  cum_bias = 0;
  total_volume = 1;

  //read input file
  read_input(input_filename);
  
}


void EDMBias::subdivide(double sublo[3], double subhi[3]) {

  //update the  grid based on the subdivision by modifying its public variables

  //call initialize, which will allocate the arrays and compute minigrid for the gaussian grid

}

void EDMBias::update_forces(int nlocal, double** positions, double** fexternal) {

  //simply perform table look-ups of the positions to get the forces

}


void EDMBias::add_hills(int nlocal, double** positions) {

  //get current hill height
  double bias_added = 0;
  double h = hill_prefactor;  
  if(global_tempering > 0)
    if(cum_bias / total_volume >= global_tempering)
      h *= exp(-(cum_bias / total_volume - global_tempering) / ((bias_factor - 1) * temperature * boltzmann_factor));                   
  double this_h;


  //for x in positions && sample random number if we're doing  hills density > 1
  {
    this_h = h;
    this_h /= exp(target.get_value(x));
    if(b_tempering && global_tempering < 0)
      this_h *= exp(-bias.get_value(x) / ((bias_factor - 1) * temperatur * boltzmann_factor));
    bias.add_gaussian(x, this_h);
    bias_added += this_h;
  }

  update_height(bias_added);

}

void EDMBias::update_height(double bias_added) {
  //reduce bias_added  MPI_reducev(bias_added, +)
  //store result in cum_bias
  
}


void EDMBias::read_input(std::string& input_filename){ 
  //parse file

  //build grid
  
  //set boundary
  total_volume = bias.get_volume();
}
