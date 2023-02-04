"""
Some default parameters and custom data types.
"""

# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

from typing import Union, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


# =============================================================================
#  GLOBAL VARIABLES
# =============================================================================

DEFAULT_FIG_SIZE = (7, 4)


# =============================================================================
#  DATA TYPES
# =============================================================================

NumLike = Union[int, float]
ArrayLike = Union[list, np.array, pd.Series, tf.Tensor]
RegressionData = Tuple[ArrayLike, ArrayLike]
HeightAndWidth = Tuple[NumLike, NumLike]
