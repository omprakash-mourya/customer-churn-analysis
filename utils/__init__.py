"""
Utilities package for Customer Churn Prediction.
"""

# Import only what works
try:
    from .preprocessing import ChurnDataPreprocessor
    PREPROCESSING_AVAILABLE = True
except ImportError:
    PREPROCESSING_AVAILABLE = False

try:
    from .metrics import ModelEvaluator
    METRICS_AVAILABLE = True  
except ImportError:
    METRICS_AVAILABLE = False

try:
    from .model_trainer import ChurnModelTrainer
    MODEL_TRAINER_AVAILABLE = True
except ImportError:
    MODEL_TRAINER_AVAILABLE = False

try:
    from .visualization import ChurnVisualizer  
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Only export what's available
__all__ = []
if PREPROCESSING_AVAILABLE:
    __all__.append('ChurnDataPreprocessor')
if METRICS_AVAILABLE:
    __all__.append('ModelEvaluator')
if MODEL_TRAINER_AVAILABLE:
    __all__.append('ChurnModelTrainer') 
if VISUALIZATION_AVAILABLE:
    __all__.append('ChurnVisualizer')