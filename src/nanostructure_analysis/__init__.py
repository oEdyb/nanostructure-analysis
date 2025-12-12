"""
Nanostructure Analysis Package

A scientific data analysis toolkit for optical characterization of gold nanostructures.
Processes and correlates spectroscopy, confocal microscopy, APD time traces, and SEM images.
"""

__version__ = "0.1.0"

# Import configuration (available as nsa.config)
from . import config

# Import all modules to make them accessible via the package
from . import apd_functions
from . import apd_plotting_functions
from . import confocal_functions
from . import confocal_plotting_functions
from . import loading_functions
from . import sem_functions
from . import sem_plotting_functions
from . import spectra_functions
from . import spectra_plotting_functions

# Import all commonly used functions to allow 'from nanostructure_analysis import *'
from .spectra_functions import *
from .confocal_functions import *
from .apd_functions import *
from .sem_functions import *
from .loading_functions import *
from .apd_plotting_functions import *
from .spectra_plotting_functions import *
from .confocal_plotting_functions import *
from .sem_plotting_functions import *

# Let modules define their own __all__ exports
# Users can do: from nanostructure_analysis import *
