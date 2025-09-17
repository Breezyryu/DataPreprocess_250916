"""
BatteryAnalyzer - Universal Battery Test Data Analysis Framework

A comprehensive framework for analyzing battery test data from various manufacturers
and test equipment, with electrochemical degradation analysis capabilities.
"""

__version__ = "1.0.0"
__author__ = "Battery Analysis Team"

from .core.data_loader import UniversalBatteryLoader
from .analyzers.cycle_analyzer import CycleAnalyzer
from .analyzers.degradation_analyzer import DegradationAnalyzer