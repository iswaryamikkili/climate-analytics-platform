"""
Weather Forecasting Module - Base Classes and Utilities
Implements foundation for multiple forecasting algorithms
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
