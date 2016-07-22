import os
from glob import glob
from setuptools import setup

def readme():
    with open('../README.md') as f:
        return f.read()

exec(open('edm/version.py').read())

setup(name = 'peptidesim', 
      version = __version__,
      description = 'Electronic Dance Music Python Bindings',
      long_description=readme(),
      author = 'Andrew White', 
      author_email = 'andrew.white@rochester.edu', 
      url = 'http://thewhitelab.org/Software',
      license = 'GPL3',
      packages = ['edm']
)
