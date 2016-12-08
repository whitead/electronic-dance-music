#ifndef EDM_H_
#define EDM_H_

#include <iostream>


#ifdef __CUDACC__
#ifndef HOST_DEV
#define HOST_DEV __host__ __device__
#endif //HOST_DEV
#else
#define HOST_DEV
#endif //CUDAACC



namespace EDM{ 

  void edm_error(const char* error, const char* location);

}

#endif //EDM_H_
