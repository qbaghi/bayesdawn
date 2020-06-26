# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path
# io.open is needed for projects that support Python 2.7
# It ensures open() defaults to text mode with universal newlines,
# and accepts an argument to specify the text encoding
# Python 3 only projects can skip this import
from io import open
# For cython extensions
import os
import numpy
from Cython.Build import cythonize
from distutils.extension import Extension

here = path.abspath(path.dirname(__file__))
print(here)
# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Extensions
extensions = [Extension("bayesdawn.waveforms.cywavefuncs",
                        ["bayesdawn/waveforms/cywavefuncs.pyx"])]

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='bayesdawn',
    version='1.0.0',
    description='a_mat bayesian data augmentation algorithm',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/qbaghi/bayesdawn',
    author='Quentin Baghi',
    author_email='quentin.baghi@protonmail.com',
    classifiers=[  # Optional
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish
        'License :: OSI Approved :: GNU License',

        # Python versions
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    keywords='bayesesian data analysis',
    packages=find_packages(),
    python_requires='>=3.5', install_requires=['cython',
                                               'h5py',
                                               'scipy',
                                               'numpy'],
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include(),
                  os.path.join(numpy.get_include(), 'numpy')]

)

