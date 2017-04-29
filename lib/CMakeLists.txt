add_definitions(-DEDM_SERIAL)

add_library (edm SHARED edm_bias grid gaussian_grid edm)


install (TARGETS edm DESTINATION lib)
install (FILES edm_bias.h gaussian_grid.h grid.h edm.h DESTINATION include/edm)






