"""A setuptools based mlak project setup module."""

from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
import re

here = path.abspath( path.dirname( __file__ ) )

# Get the long description from the README file
with open( path.join( here, "README.rst" ), encoding = "utf-8" ) as f:
	long_description = f.read()

with open( path.join( here, "mlak/__init__.py" ), encoding = "utf-8" ) as f:
	# Versions should comply with PEP 440: https://www.python.org/dev/peps/pep-0440/
	version = re.search( r"^__version__\s*=\s*['\"]([^'\"]*)['\"]", f.read(), re.M )
	if not version:
		raise RuntimeError( "Unable to find version string." )
	version = version.group( 1 )

setup(
	name = "mlak",
	version = version,
	description = "Machine Learning Army Knife",
	long_description = long_description,
	url = 'https://github.com/codestation-org/mlak',
	author = "CodeStation.org",
	author_email = "mlak@codestation.org",
	maintainer = "Marcin `amok` Konarski",
	maintainer_email = "amok@codestation.org",
	# Classifiers help users find your project by categorizing it.
	# For a list of valid classifiers, see: https://pypi.python.org/pypi?%3Aaction=list_classifiers
	classifiers = [  # Optional
		# How mature is this project? Common values are: 3 - Alpha, 4 - Beta, 5 - Production/Stable
		"Environment :: Console",
		"Development Status :: 3 - Alpha",
		"Intended Audience :: Science/Research",
		"Topic :: Scientific/Engineering :: Artificial Intelligence",
		# Pick your license as you wish
		"License :: OSI Approved :: Python Software Foundation License",
		"Programming Language :: Python :: 3",
	],
	license = "Python Software Foundation License",
	keywords = "machine learning keras neural network neuronal logistic regression linear model analyzer",  # Optional
	packages = find_packages( exclude = ['contrib', 'docs', 'tests'] ),  # Required
	install_requires = ["numpy", "scipy", "matplotlib", "tensorflow", "keras"],  # Optional

	extras_require = {  # Optional
		'test': ['coverage'],
	},
	test_suite = "unittest2.collector",

	# If there are data files included in your packages that need to be
	# installed, specify them here.
	package_data={  # Optional
		'sample': ['package_data.dat'],
	},
	entry_points = {  # Optional
		'console_scripts': [
			'mlak = mlak.mlak:main',
		],
	},
)

