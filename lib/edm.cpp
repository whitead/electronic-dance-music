#include "edm.h"
#include <stdlib.h>

void EDM::edm_error(const char* error, const char* location) {
  std::cerr << "[EDM:" << error << "] " << location << std::endl;
  abort();
}
