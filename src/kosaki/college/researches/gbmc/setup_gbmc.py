
from setuptools import setup, Extension
module = Extension('gbmc_core', sources=['gbmc_core.c'])
setup(
    name='gbmc_core',
    version='1.0',
    ext_modules=[module]
)
