import libedm_python
import random

class EDMBias(libedm_python.EDMBias_Py):
    def add_hill(self, position):
        self.pre_add_hill(1)
        self.add_hill_r(position, random.random())
        self.post_add_hill()
