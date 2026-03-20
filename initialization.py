import importlib.util
import subprocess
import sys

required_packages = {
    "wfdb": "wfdb",
    "xgboost": "xgboost",
    "tensorflow": "tensorflow",
}

for module_name, package_name in required_packages.items():
    if importlib.util.find_spec(module_name) is None:
        print(f"Installing missing package: {package_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_name])

import gc
import os
import random
import re
import sys
import traceback
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import wfdb

from IPython.display import display
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
from scipy.stats import kurtosis, skew
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models, optimizers
from tqdm.auto import tqdm
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

randomState = 42
fsExpected = 100
standardLeads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 4)

projectRoot = Path.cwd()
datasetDir = projectRoot / "datasets"
outputRoot = projectRoot / "outputs"
plotDir = outputRoot / "plots"
predictionDir = outputRoot / "predict"
logFile = outputRoot / "logs.txt"
outputRoot.mkdir(exist_ok=True)
plotDir.mkdir(exist_ok=True)
predictionDir.mkdir(exist_ok=True)

class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

if "_original_stdout" not in globals():
    _original_stdout = sys.stdout
if "_original_stderr" not in globals():
    _original_stderr = sys.stderr
if "_log_handle" in globals() and getattr(_log_handle, "closed", True) is False:
    _log_handle.close()

_log_handle = open(logFile, "w", encoding="utf-8", buffering=1)
sys.stdout = TeeStream(_original_stdout, _log_handle)
sys.stderr = TeeStream(_original_stderr, _log_handle)

_saved_plot_counts = {}

def sanitize_name(text):
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip()).strip("_").lower()
    return text or "artifact"

def move_legends_outside(fig=None):
    fig = fig or plt.gcf()
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is None:
            continue
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            continue
        legend_title = legend.get_title().get_text() if legend.get_title() is not None else None
        legend.remove()
        ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0.0,
            title=legend_title if legend_title else None,
        )

def save_plot(plot_name):
    fig = plt.gcf()
    move_legends_outside(fig)
    safe_name = sanitize_name(plot_name)
    current_count = _saved_plot_counts.get(safe_name, 0) + 1
    _saved_plot_counts[safe_name] = current_count
    suffix = "" if current_count == 1 else f"_{current_count:02d}"
    file_path = plotDir / f"{safe_name}{suffix}.png"
    fig.savefig(file_path, dpi=300, bbox_inches="tight")
    print("Saved plot:", file_path)
    plt.show()
    plt.close(fig)
    return file_path

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_global_seed(randomState)
print("Workspace root:", projectRoot)
print("Dataset directory:", datasetDir)
print("Output root:", outputRoot)
print("Plot directory:", plotDir)
print("Prediction directory:", predictionDir)
print("Log file:", logFile)

datasetRoot = datasetDir
metaPath = datasetRoot / "metadata.csv"
assert metaPath.exists(), (
    f"metadata.csv was not found at {metaPath}. Put the Brugada HUCA dataset inside the local datasets folder and rerun this cell."
)

metadata = pd.read_csv(metaPath)

requiredMetadataColumns = {"patient_id", "basal_pattern", "sudden_death", "brugada"}
missingMetadataColumns = requiredMetadataColumns.difference(metadata.columns)
assert not missingMetadataColumns, (
    f"Missing required metadata columns: {sorted(missingMetadataColumns)}"
)

metadata["brugada"] = pd.to_numeric(metadata["brugada"], errors="coerce")
classCounts = metadata["brugada"].value_counts(dropna=False).to_dict()
expectedTotalSubjects = 363
expectedBrugadaSubjects = 76
expectedNormalSubjects = 287

observedTotalSubjects = len(metadata)
observedBrugadaSubjects = int((metadata["brugada"] > 0).sum())
observedNormalSubjects = int((metadata["brugada"] == 0).sum())

datasetSpecChecks = [
    ("Total subjects", observedTotalSubjects, expectedTotalSubjects),
    ("Brugada-positive subjects (raw label > 0)", observedBrugadaSubjects, expectedBrugadaSubjects),
    ("Normal subjects", observedNormalSubjects, expectedNormalSubjects),
]
datasetSpecIssues = []
for label, observed, expected in datasetSpecChecks:
    status = "OK" if observed == expected else "WARNING"
    print(f"{label}: observed={observed}, expected={expected} [{status}]")
    if observed != expected:
        datasetSpecIssues.append(f"{label} mismatch: observed {observed}, expected {expected}.")

if datasetSpecIssues:
    print("Dataset specification cross-check warnings:")
    for issue in datasetSpecIssues:
        print("-", issue)
    print("Execution will continue with the detected metadata values.")
else:
    print("Dataset specification cross-check passed.")

print("Metadata path:", metaPath)
print("Dataset root:", datasetRoot)
print("Metadata shape:", metadata.shape)
print("Metadata columns:", metadata.columns.tolist())
print("Raw class counts:", classCounts)
display(metadata.head())
