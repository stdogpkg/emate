from setuptools import setup, find_packages
import ctypes

# to check pip version
import pkg_resources 

pipVersion = pkg_resources.require("pip")[0].version
setuptoolsVersion = pkg_resources.require("setuptools")[0].version

print("\n PIP Version", pipVersion, "\n")
print("\n Setuptools Version", setuptoolsVersion, "\n")

olderPip = pipVersion < "20.0"
olderSetuptools = setuptoolsVersion < "45.0"

def checkCUDAisAvailable():
    """
    This function check if any of this possible libs are available.
    see https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549

    Returns:
    --------
        libsOk : bool
            If True then CUDA is available
    """
    # some possible lib names 
    libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
    libsOk = True
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        libsOk = False
    return libsOk

def getRequirements():
    """
    This function it's used in order to get the package names. Which
    depends on the libs available in the machine. 

    Return:
    -------
        conditionalRequirements: list
            A list of strings containing the pip pkgs.
    """

 
    cudaLibsOk = checkCUDAisAvailable()   
    
    conditionalRequirements = []
    if cudaLibsOk:
        conditionalRequirements += ["tensorflow-gpu==1.15.3", ]
    else:
        print("\n CUDA it's not available in your machine.")
        print(" You won't be able to use the GPU support.\n")
        #if olderPip or olderSetuptools:
        #tfRequirement = "tensorflow==1.15.0"
        #else:
        tfRequirement = "tensorflow==1.15.3"
    
        conditionalRequirements += [tfRequirement]

    return conditionalRequirements

conditionalRequirements = getRequirements()
install_requires = ["scipy", "numpy"] + conditionalRequirements

with open("README.md", "r") as f:
    README_TEXT = f.read()

setup(
    name="emate",
    version="v1.1.3",
    packages=find_packages(exclude=["build", ]),
    long_description=README_TEXT,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    include_package_data=True,
    license="MIT",
    description="""eMaTe can run in both CPU and GPU and can 
        estimate the spectral density and related trace functions, 
        such as entropy and Estrada index, even in matrices 
        (directed or undirected graphs) with 
        million of nodes.""",
    author_email="messias.physics@gmail.com",
    author="Bruno Messias; Thomas K Peron",
    download_url="https://github.com/stdogpkg/emate/archive/v1.0.4.tar.gz",
    keywords=[
        "gpu", "science", "complex-networks", "graphs", "matrices", "kpm",
         "tensorflow", "chebyshev", "spectral", "eigenvalues"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
    url="https://github.com/stdogpkg/emate"
)
