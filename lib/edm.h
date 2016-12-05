#include <iostream>
#define EDM_GPU_MODE_NO

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
