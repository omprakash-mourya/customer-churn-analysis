#!/usr/bin/env python3
"""
Test utils module imports
"""

import sys
import os

# Add utils to path
sys.path.insert(0, 'utils')

print("Testing utils imports...")

try:
    from utils.preprocessing import ChurnDataPreprocessor
    print("✅ Preprocessing: OK")
    PREPROCESSING_OK = True
except Exception as e:
    print(f"❌ Preprocessing: {e}")
    PREPROCESSING_OK = False

try:
    from utils.visualization import ChurnVisualizer
    print("✅ Visualization: OK")
    VISUALIZATION_OK = True
except Exception as e:
    print(f"❌ Visualization: {e}")
    VISUALIZATION_OK = False

try:
    from utils.metrics import ModelEvaluator
    print("✅ Metrics: OK")
    METRICS_OK = True
except Exception as e:
    print(f"❌ Metrics: {e}")
    METRICS_OK = False

try:
    from utils.model_trainer import ChurnModelTrainer
    print("✅ Model Trainer: OK")
    TRAINER_OK = True
except Exception as e:
    print(f"❌ Model Trainer: {e}")
    TRAINER_OK = False

overall_status = PREPROCESSING_OK and VISUALIZATION_OK and METRICS_OK
print(f"\n🎯 Overall Utils Status: {'✅ OK' if overall_status else '❌ ISSUES'}")

if not overall_status:
    print("\n🔧 Fix needed for utils to work properly")
else:
    print("\n🎉 All utils modules working!")
