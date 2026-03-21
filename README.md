# **IDSC2026 Brugada-HUCA Classification**

<p align="left">
  <img src="https://img.shields.io/badge/VZ%20Team-16a34a?style=for-the-badge&logoColor=white" alt="VZ Team" />
  
  <a href="https://www.kaggle.com/username" target="_blank">
    <img src="https://img.shields.io/badge/Open%20in%20Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Open in Kaggle" />
  </a>
  
  <a href="https://github.com/avezoor/IDSC2026" target="_blank">
    <img src="https://img.shields.io/badge/Repository-181717?style=for-the-badge&logo=github&logoColor=white" alt="Repository" />
  </a>
</p>

# 1. Installation

Install the dependencies from [requirements.txt](requirements.txt) before running the pipeline.

## 1.1 Create Virtual Environment

This step is optional, but recommended so the project dependencies do not mix with your global environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

If you are using Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## 1.2 Install Dependencies From `requirements.txt`

Run the following commands from the project root:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 1.3 Run The Project

After all packages are installed, run:

```bash
python main.py
```

## 1.4 Main Outputs After Running

After `python main.py`, the pipeline writes readable output filenames so the main artifacts are easy to inspect:

- `outputs/Summary.csv`: leaderboard-oriented ECG-only benchmark summary.
- `outputs/Clinical Summary.csv`: safety-oriented clinical ranking with stronger emphasis on sensitivity and false negatives.
- `outputs/Multi Aspect Ranking.csv`: balanced ranking that combines performance, safety, efficiency, and stability evidence.
- `outputs/Pareto Front Models.csv`: models that remain competitive on the performance / safety frontier.
- `outputs/Best Models.txt`: short final recommendation text.
- `outputs/Logs.txt`: end-to-end run log.
- `outputs/plots/`: EDA, validation, interpretability, and failure-analysis figures.
- `outputs/predict/`: per-model predictions, split files, metadata ablation exports, and stability summaries.
- `Tech Report.tex` and `Tech Report.pdf`: report source and the compiled PDF synced to the latest rerun.

**Current recommendation snapshot**

- Balanced multi-aspect recommendation: `Transfer Learning (In-domain DAE to Classifier)`.
- Safety-oriented clinical candidate: `ResNet 1D Median Beat`.

# 2. Overview

**Objective**

Classify Brugada Syndrome versus Normal ECG recordings from the Brugada-HUCA dataset using patient-level ECG benchmarking with explicit attention to class imbalance, clinical safety, interpretability, and responsible deployment.

**Strategy Pipeline**

1. Locate the dataset automatically and validate the expected recording structure, metadata, 12-lead coverage, 100 Hz sampling, and 12-second duration.
2. Explore label balance and metadata distributions while keeping the final primary benchmark focused on ECG evidence rather than shortcut-risk clinical metadata.
3. Build explicit preprocessing, engineered ECG features, and median-beat sequence representations from the raw WFDB records.
4. Create one shared stratified patient split so every model is compared on the same train, validation, and test partitions.
5. Use `ECG-only` as the primary benchmark and reserve `ECG + metadata` for a separate ablation so shortcut learning can be discussed transparently without contaminating the main comparison.
6. Train 11 ECG-only models across feature-based, deep learning, and transfer-learning families with imbalance-aware weighting and validation-based threshold tuning.
7. Report both a `leaderboard-oriented` ranking and a `safety-oriented clinical` ranking so aggregate performance is not confused with the best low-false-negative clinical story.
8. Add metadata ablation, repeated split stability checks, threshold trade-off analysis, interpretability, failure analysis, and a limitations / responsible-deployment section.

**Why this approach**

- All models below use the same patient split data so the direct benchmark stays fair.
- The pipeline checks that the dataset matches the expected cohort structure: 363 subjects, 76 Brugada cases, 287 Normal controls, 12 leads, 100 Hz, and 12-second records.
- The Brugada class is the minority class, so the evaluation emphasizes PR-AUC, F1, sensitivity, balanced accuracy, threshold analysis, and Brugada case detection instead of raw accuracy alone.
- The primary benchmark is ECG-only. Metadata such as `basal_pattern` and `sudden_death` are analyzed separately because they may act as shortcut clinical signals rather than pure ECG evidence.
- This project uses ECG-only as the primary benchmark to avoid shortcut learning from metadata and to keep the main model comparison fair, clinically defensible, and genuinely ECG-driven.
- The final outputs report both the aggregate winner and the safety-oriented clinical candidate because the model with the best overall ranking may still be suboptimal for a Brugada screening narrative if too many true Brugada cases are missed.
- The dataset is loaded as raw ECG data, so filtering, normalization, median-beat extraction, and feature engineering remain explicit pipeline preprocessing choices.
- Interpretability is included through feature importance and sequence saliency, with special attention to the right precordial leads `V1-V3`, but these outputs are treated as descriptive transparency rather than proof of causal reasoning.
- Repeated split stability analysis is included to strengthen validation rigor, although the project still acknowledges that one held-out test split across many candidate models remains vulnerable to model-selection optimism.
- Any model selected here should be discussed as research-oriented decision support, not as an autonomous clinical diagnostic system.

**References / Citation**

When using this resource, please cite:

Costa Cortez, N., & Garcia Iglesias, D. (2026). *Brugada-HUCA: 12-Lead ECG Recordings for the Study of Brugada Syndrome* (version 1.0.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/0m2w-dy83

Please include the standard citation for PhysioNet:

Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). *PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals*. Circulation [Online]. 101 (23), pp. e215-e220. RRID:SCR_007345.

# 3. Initialization

This section prepares the dependencies, the base configuration, and the automatic dataset discovery logic.

## 3.1 Install Required Packages

Install the required packages only when they are missing from the current environment.

## 3.2 Import Libraries and Prepare Local Output Folders

Import the core libraries, set the random seed, and prepare the local `outputs/plots/` and `outputs/predict/` folders for artifacts generated by the pipeline.

## 3.3 Locate Dataset Automatically

Search for `metadata.csv` across several candidate locations and select the most likely dataset root automatically.

# 4. EDA

This section checks the label distribution, ECG file availability, and a few initial signal examples.

## 4.1 Inspect Label and Metadata Distribution

Review the Brugada vs Normal class balance and the distribution of key metadata fields. This step also highlights the class imbalance that must be handled during training and validation.

## 4.2 Index ECG Record Files

Build a WFDB file index so each `patient_id` can be matched to its ECG record.

## 4.3 Define ECG Loading Helper

Create a helper function to load ECG records and standardize the lead names.

## 4.4 Visualize Example ECG Signals

Display example signals from one Normal patient and one Brugada patient for a quick visual check.

# 5. Preprocessing

This section cleans the signal, extracts the median beat, and builds both the feature-based and sequence-based representations.

## 5.1 Define Signal Preprocessing Helpers

Create the filtering, robust normalization, R-peak detection, and median beat extraction helpers.

Important note: the dataset is loaded as raw ECG data, and the filtering and normalization below are explicit pipeline preprocessing choices added after loading the raw WFDB records.

## 5.2 Define Feature Engineering Helpers

Create the strip-level statistical features and the beat-level morphology features.

## 5.3 Build Feature Table

Run preprocessing for every patient and combine the outputs into one feature table.

# 6. Test Split

This section builds one shared patient split for all models, defines the ECG-only primary feature set, and prepares the metadata-inclusive ablation set separately.

## 6.1 Create the Shared Patient Split

Split the data into train, validation, and test sets at the patient level with stratification, then prepare the tabular and sequence inputs. Stratification is required here because the Brugada class is the minority class.

## 6.2 Visualize the Shared Split

Display the class distribution in each split to confirm that the split remains balanced.

# 7. Train Models

All models below use the same patient split. The primary benchmark in this section is ECG-only. Metadata columns remain excluded from the main training benchmark and are only revisited later as a separate shortcut-risk ablation. This project does not use metadata in the main model comparison because that could make the model look stronger for the wrong reason. Because the dataset is imbalanced, the training setup uses imbalance-aware weighting, threshold tuning, and validation metrics that do not rely on accuracy alone.

## 7.1 Feature-Based Models

The models in this section use manually engineered ECG features.

### 7.1.1 Define Evaluation and Export Helpers

Create the helper functions for threshold tuning, metric evaluation, and prediction export. These helpers are designed to keep the benchmark imbalance-aware by focusing on PR-AUC, F1, sensitivity, balanced accuracy, and class-level Brugada detection.

### 7.1.2 Train Feature-Based Models

Train `XGBoost`, `Random Forest`, `SVM RBF`, and `AdaBoost` on the same feature table. This step uses imbalance-aware settings such as `scale_pos_weight`, balanced class weights, weighted AdaBoost fitting, and validation-based threshold tuning.

## 7.2 Deep Learning End-to-End Models

The models in this section learn directly from the 12-lead median beat ECG representation.

### 7.2.1 Define Deep Learning Architectures

Create the `1D CNN`, `ResNet 1D`, `CNN + BiGRU`, and `Transformer Encoder` architectures.

### 7.2.2 Define the Deep Training Routine

Create one shared training routine so every deep learning model follows a consistent workflow.

### 7.2.3 Train Deep Learning Models

Train all four deep learning models on the same median beat split.

## 7.3 Transfer Learning and Advanced Models

This section focuses on advanced strategies that help when Brugada data is limited.

### 7.3.1 Define In-Domain Transfer Learning Helpers

Create the encoder, denoising autoencoder, and classifier used by the in-domain transfer learning pipeline.

### 7.3.2 Define VICReg Self-Supervised Helpers

Create the `VICReg` components for label-free pretraining on the same Brugada HUCA data.

### 7.3.3 Define Echo State Network Helper

Create the reservoir transformation and classifier pipeline for the `ESN` model.

### 7.3.4 Train Transfer Learning and Advanced Models

Train `Transfer Learning`, `VICReg`, and `ESN`, then save the prediction output for each model.

# 8. Validation

This section ranks the models, compares the metrics, and adds supporting validation analyses. Because the dataset is imbalanced, the pipeline reports both a leaderboard-oriented ranking and a safety-oriented ranking that gives earlier priority to Brugada sensitivity and false-negative control. The main comparison still relies on one held-out test split across many models, so the results should be interpreted with explicit caution about model-selection optimism.

## 8.1 Build Leaderboards and Save Summary Files

Combine all model results, save the standard leaderboard summary, and also build a safety-oriented clinical ranking. The first ranking is still useful for aggregate benchmarking, while the second makes it explicit when the most competition-friendly model is not the same as the most defensible low-false-negative candidate. This distinction is especially important here because a model can lead on PR-AUC or F1 while still missing too many Brugada cases for a strong clinical-safety narrative.

## 8.2 Create Validation Comparison Plots

Build the comparison plots for performance metrics, ROC curves, PR curves, and per-class counts. The visual focus remains on metrics that are informative under class imbalance.

## 8.3 Inspect the Safety-Oriented Clinical Candidate

Display the confusion matrix for the safety-oriented clinical candidate, compare it with the leaderboard winner, and summarize the training history when available. The goal is to make the sensitivity versus false-positive trade-off visible instead of assuming the leaderboard winner is automatically the best model for the clinical story.

## 8.4 Metadata Leakage Ablation

Re-run the feature-based models with `ECG + metadata` so the effect of shortcut-risk columns can be measured explicitly instead of being mixed into the primary benchmark. This analysis is supplementary only and must not replace the ECG-only benchmark.

## 8.5 Repeated Stratified Stability Check

Use repeated patient-level stratified splits on the train-plus-validation pool for the most practical ECG-only finalists. This does not replace external validation, but it gives a stronger sense of how unstable the results may be on a small dataset.

## 8.6 Interpretability and Failure Analysis

Add model explanations that can be shown to judges: permutation importance for ECG features, gradient saliency for the strongest saved sequence model, and a focused false-negative analysis for the safety-oriented clinical candidate. These analyses are intended to strengthen clinical plausibility, especially around V1-V3, but they should still be treated as descriptive transparency rather than definitive proof of mechanistic reasoning.

## 8.7 Limitations and Responsible Deployment

- The primary benchmark is ECG-only. `basal_pattern` and `sudden_death` are kept out of the main model comparison because they can act as shortcut-risk clinical metadata.
- If metadata improves the score, that result should be interpreted as a shortcut-risk sensitivity analysis, not as evidence that the primary ECG model is genuinely better.
- The dataset is small, imbalanced, and single-center. Even with repeated patient-level split checks, the variance of the results can still be high.
- The project compares many models on one held-out test split, so the reported ranking is still vulnerable to model-selection optimism.
- The leaderboard winner is not automatically the strongest clinical choice. A model can rank first on aggregate metrics while still having a sensitivity that is difficult to defend for Brugada screening.
- False negatives remain a major safety concern in Brugada screening. Any candidate model should be discussed as decision support, not as an autonomous diagnostic replacement.
- Threshold choice should be calibrated prospectively and re-validated externally before any real deployment claim is made.
- Interpretability plots in this project are descriptive analyses for transparency, not proof of causal clinical reasoning. Their role is to support a clinically coherent argument, not to replace prospective validation or electrophysiology expertise.

# 9. Summary

This final section prints both the leaderboard winner and the safety-oriented clinical candidate. The final report keeps those two perspectives separate so the competition-facing result does not hide a poor false-negative profile.

## 9.1 Print Final Decision and Log File

Display the leaderboard winner, the safety-oriented clinical candidate, the main safety notes, and the location of all output files. Then print the content of `Logs.txt`.
