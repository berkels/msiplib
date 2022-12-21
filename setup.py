from setuptools import setup, find_packages

setup(
    name='msiplib',
    packages=find_packages(),
    description='A collection of functions and classes useful for signal and image processing.',
    version='0.1',
    url='https://git.rwth-aachen.de/msip/msiplib',
    author='Benjamin Berkels',
    author_email='berkels@aices.rwth-aachen.de',
    install_requires=[
        'imageio',
        'joblib',
        'netCDF4',
        'ncempy',
        'numba',
        'pyfftw',
        'pyyaml',
        'scipy',
        'scikit-image',
        'scikit-learn',
        'tqdm'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    )
