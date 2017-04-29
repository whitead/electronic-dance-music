# Install/unInstall package files in LAMMPS
# Since we rely on the EDM library being installed locally, this should be all we need to do.

if (test $1 = 1) then

  cp fix_edm.cpp ..
  cp fix_edm.h ..
  cp fix_edm_pair.cpp ..
  cp fix_edm_pair.h ..

elif (test $1 = 0) then
  rm -f ../fix_edm.cpp
  rm -f ../fix_edm.h
  rm -f ../fix_edm_pair.cpp
  rm -f ../fix_edm_pair.h
fi
