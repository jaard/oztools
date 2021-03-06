import sys
from setuptools import setup, find_packages

MAJOR = 0
MINOR = 1
MICRO = 0
ISRELEASED = False
VERSION = "%d.%d.%d" % (MAJOR, MINOR, MICRO)
QUALIFIER = ""


DISTNAME = "oztools"
LICENSE = "MIT"
AUTHOR = "Jaard Hauschildt"
AUTHOR_EMAIL = "jhauschildt@geomar.de"
URL = "https://github.com/jaard/oztools"
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 2.6",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.3",
    "Programming Language :: Python :: 3.4",
    "Topic :: Scientific/Engineering",
]

INSTALL_REQUIRES = ["pandas", "numpy", "scipy", "matplotlib", "seawater", "gsw"]
SETUP_REQUIRES = ["pytest-runner"]
TESTS_REQUIRE = ["pytest >= 2.8", "coverage"]

if sys.version_info[:2] < (2, 7):
    TESTS_REQUIRE += ["unittest2 == 0.5.1"]


DESCRIPTION = "Tools to analyse observational ocean data"
with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

setup(
    name=DISTNAME,
    version=VERSION,
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    classifiers=CLASSIFIERS,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    install_requires=INSTALL_REQUIRES,
    setup_requires=SETUP_REQUIRES,
    tests_require=TESTS_REQUIRE,
    url=URL,
    packages=find_packages(),
)
