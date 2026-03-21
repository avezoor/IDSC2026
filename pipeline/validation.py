assert benchmarkResults, "No model finished successfully, so the leaderboard cannot be built. Check Logs.txt for the failure trace."

benchmarkResultsDf = pd.DataFrame(benchmarkResults)
benchmarkResultsDf = benchmarkResultsDf.sort_values(
    by=leaderboardSortColumns,
    ascending=[False] * len(leaderboardSortColumns),
).reset_index(drop=True)
benchmarkResultsDf.insert(0, "Rank", np.arange(1, len(benchmarkResultsDf) + 1))

clinicalResultsDf = pd.DataFrame(benchmarkResults)
clinicalResultsDf = clinicalResultsDf.sort_values(
    by=clinicalSortColumns,
    ascending=clinicalAscending,
).reset_index(drop=True)
clinicalResultsDf.insert(0, "Clinical Rank", np.arange(1, len(clinicalResultsDf) + 1))

benchmarkCountDf = pd.concat(benchmarkCountReports, axis=0).reset_index(drop=True)
benchmarkFailuresDf = pd.DataFrame(benchmarkFailures)

summaryFile = outputRoot / make_filename("summary", "csv")
clinicalSummaryFile = outputRoot / make_filename("clinical summary", "csv")
benchmarkResultsDf.to_csv(summaryFile, index=False)
clinicalResultsDf.to_csv(clinicalSummaryFile, index=False)

countSummaryFile = predictionDir / make_filename("model count comparison", "csv")
benchmarkCountDf.to_csv(countSummaryFile, index=False)

failedModelsFile = predictionDir / make_filename("failed models", "csv")
benchmarkFailuresDf.to_csv(failedModelsFile, index=False)

print("Saved leaderboard summary:", repoDisplayPath(summaryFile))
print("Saved safety-oriented clinical summary:", repoDisplayPath(clinicalSummaryFile))
print("Saved class count summary:", repoDisplayPath(countSummaryFile))
print("Saved failure summary:", repoDisplayPath(failedModelsFile))
print("Leaderboard priority:", " -> ".join(leaderboardSortColumns))
print("Clinical priority:", " -> ".join(clinicalSortColumns))
print("Successful models:", len(benchmarkResultsDf))
print("Failed models:", len(benchmarkFailuresDf))
display(benchmarkResultsDf)
display(clinicalResultsDf)
display(benchmarkCountDf)
display(benchmarkFailuresDf)

plt.figure(figsize=(14, 5))
sns.barplot(data=benchmarkResultsDf, x="Model", y="Correctly Predicted", hue="Family")
plt.title("Correct Predictions by Model")
plt.xlabel("")
plt.ylabel("Correctly predicted")
plt.xticks(rotation=30, ha="right")
save_plot("validation_correct_predictions_by_model")

metricHeatmapDf = benchmarkResultsDf.set_index("Model")[["AUC", "PR AUC", "F1", "Accuracy", "Sensitivity", "Specificity"]]
plt.figure(figsize=(10, 6))
sns.heatmap(metricHeatmapDf, annot=True, cmap="YlGnBu", vmin=0.0, vmax=1.0)
plt.title("Metric Heatmap Across Models")
save_plot("validation_metric_heatmap")

plt.figure(figsize=(10, 7))
for modelName, predDf in benchmarkPredictionTables.items():
    fpr, tpr, _ = roc_curve(predDf["target"], predDf["pred_proba"])
    aucValue = roc_auc_score(predDf["target"], predDf["pred_proba"])
    plt.plot(fpr, tpr, label=f"{modelName} (AUC={aucValue:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve Comparison on the Shared Test Split")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
save_plot("validation_roc_curve_comparison")

plt.figure(figsize=(10, 7))
for modelName, predDf in benchmarkPredictionTables.items():
    precisionCurve, recallCurve, _ = precision_recall_curve(predDf["target"], predDf["pred_proba"])
    prAucValue = average_precision_score(predDf["target"], predDf["pred_proba"])
    plt.plot(recallCurve, precisionCurve, label=f"{modelName} (PR-AUC={prAucValue:.3f})")
plt.title("Precision Recall Curve Comparison on the Shared Test Split")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
save_plot("validation_pr_curve_comparison")

countPlotDf = benchmarkCountDf[benchmarkCountDf["Class"] != "TOTAL"].copy()
plt.figure(figsize=(14, 5))
sns.barplot(data=countPlotDf, x="Model", y="Correctly Predicted", hue="Class")
plt.title("Correct Predictions per Class and Model")
plt.xlabel("")
plt.ylabel("Correctly predicted")
plt.xticks(rotation=30, ha="right")
save_plot("validation_correct_predictions_per_class")

leaderboardRow = benchmarkResultsDf.iloc[0]
clinicalRow = clinicalResultsDf.iloc[0]

clinicalModelName = clinicalRow["Model"]
clinicalPredictionDf = benchmarkPredictionTables[clinicalModelName]
clinicalCountDf = benchmarkCountDf[benchmarkCountDf["Model"] == clinicalModelName].copy()

clinicalConfusion = confusion_matrix(clinicalPredictionDf["target"], clinicalPredictionDf["pred_label"], labels=[0, 1])
plt.figure(figsize=(5, 4))
sns.heatmap(clinicalConfusion, annot=True, fmt="d", cmap="Blues")
plt.title(f"Clinical Candidate Confusion Matrix: {clinicalModelName}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
save_plot("validation_clinical_candidate_confusion_matrix")

comparisonDf = pd.DataFrame(
    [
        {
            "Selection Track": "Leaderboard winner",
            "Model": leaderboardRow["Model"],
            "PR AUC": leaderboardRow["PR AUC"],
            "F1": leaderboardRow["F1"],
            "Sensitivity": leaderboardRow["Sensitivity"],
            "False Negatives": leaderboardRow["False Negatives"],
            "Correct Brugada": leaderboardRow["Correct Brugada"],
        },
        {
            "Selection Track": "Safety-oriented clinical candidate",
            "Model": clinicalRow["Model"],
            "PR AUC": clinicalRow["PR AUC"],
            "F1": clinicalRow["F1"],
            "Sensitivity": clinicalRow["Sensitivity"],
            "False Negatives": clinicalRow["False Negatives"],
            "Correct Brugada": clinicalRow["Correct Brugada"],
        },
    ]
)

deepHistoryModels = [
    modelName
    for modelName, historyObj in benchmarkHistories.items()
    if isinstance(historyObj, dict) and ("loss" in historyObj or "finetune" in historyObj)
]
if deepHistoryModels:
    plt.figure(figsize=(12, 5))
    for modelName in deepHistoryModels:
        historyObj = benchmarkHistories[modelName]
        if "val_auc" in historyObj:
            val_auc_values = historyObj.get("val_auc", [])
        else:
            val_auc_values = historyObj.get("finetune", {}).get("val_auc", [])
        if val_auc_values:
            plt.plot(val_auc_values, label=modelName)
    plt.title("Validation AUC During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Validation AUC")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    save_plot("validation_training_auc_history")

display(comparisonDf)
display(clinicalPredictionDf.head(20))
display(clinicalCountDf)

metadataAblationRows = []

if metadataShortcutColumns:
    xTrainMetaDf, yTrainMeta = getTabularSplit(trainIds, includeMetadata=True)
    xValMetaDf, yValMeta = getTabularSplit(valIds, includeMetadata=True)
    xTestMetaDf, yTestMeta = getTabularSplit(testIds, includeMetadata=True)

    metadataImputer = SimpleImputer(strategy="median")
    xTrainMeta = metadataImputer.fit_transform(xTrainMetaDf)
    xValMeta = metadataImputer.transform(xValMetaDf)
    xTestMeta = metadataImputer.transform(xTestMetaDf)

    metadataScaler = StandardScaler()
    xTrainMetaScaled = metadataScaler.fit_transform(xTrainMeta)
    xValMetaScaled = metadataScaler.transform(xValMeta)
    xTestMetaScaled = metadataScaler.transform(xTestMeta)

    for modelName, baseModel in classicalModels.items():
        model = clone(baseModel)
        if modelName == "XGBoost Features":
            scalePosWeightMeta = (len(yTrainMeta) - yTrainMeta.sum()) / max(yTrainMeta.sum(), 1)
            model.set_params(scale_pos_weight=scalePosWeightMeta)

        if "SVM" in modelName:
            xTrainInput, xValInput, xTestInput = xTrainMetaScaled, xValMetaScaled, xTestMetaScaled
        else:
            xTrainInput, xValInput, xTestInput = xTrainMeta, xValMeta, xTestMeta

        if modelName == "AdaBoost":
            sampleWeightsMeta = makeSampleWeightVector(yTrainMeta)
            model.fit(xTrainInput, yTrainMeta, sample_weight=sampleWeightsMeta)
        else:
            model.fit(xTrainInput, yTrainMeta)

        valProba = model.predict_proba(xValInput)[:, 1].astype(float)
        bestThreshold, bestValF1 = tuneThreshold(yValMeta, valProba)
        testProba = model.predict_proba(xTestInput)[:, 1].astype(float)
        metricRow, _ = calculateBinaryMetrics(yTestMeta, testProba, bestThreshold)
        metricRow.update(
            {
                "Model": modelName,
                "Family": "Feature-Based",
                "Feature Set": "ECG + metadata ablation",
                "Validation F1": float(bestValF1),
            }
        )
        metadataAblationRows.append(metricRow)

    metadataAblationResultsDf = pd.DataFrame(metadataAblationRows).sort_values(
        by=["PR AUC", "F1", "Sensitivity"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    metadataAblationFile = predictionDir / make_filename("metadata ablation summary", "csv")
    metadataAblationResultsDf.to_csv(metadataAblationFile, index=False)

    ecgOnlyFeatureDf = benchmarkResultsDf[benchmarkResultsDf["Family"] == "Feature-Based"].copy()
    metadataAblationComparisonDf = ecgOnlyFeatureDf[
        ["Model", "PR AUC", "F1", "Sensitivity", "Specificity", "False Negatives", "Correct Brugada"]
    ].merge(
        metadataAblationResultsDf[
            ["Model", "PR AUC", "F1", "Sensitivity", "Specificity", "False Negatives", "Correct Brugada"]
        ],
        on="Model",
        suffixes=(" ECG-only", " ECG+metadata"),
    )
    for metricName in ["PR AUC", "F1", "Sensitivity", "Specificity", "Correct Brugada"]:
        metadataAblationComparisonDf[f"Delta {metricName}"] = (
            metadataAblationComparisonDf[f"{metricName} ECG+metadata"]
            - metadataAblationComparisonDf[f"{metricName} ECG-only"]
        )
    metadataAblationComparisonFile = predictionDir / make_filename("metadata ablation comparison", "csv")
    metadataAblationComparisonDf.to_csv(metadataAblationComparisonFile, index=False)

    print("Shortcut-risk metadata columns:", metadataShortcutColumns)
    print("Saved metadata ablation summary:", repoDisplayPath(metadataAblationFile))
    print("Saved metadata ablation comparison:", repoDisplayPath(metadataAblationComparisonFile))
    display(metadataAblationResultsDf)
    display(metadataAblationComparisonDf)

    ablationPlotDf = metadataAblationComparisonDf.melt(
        id_vars=["Model"],
        value_vars=["Delta PR AUC", "Delta F1", "Delta Sensitivity", "Delta Specificity"],
        var_name="Metric",
        value_name="Delta",
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=ablationPlotDf, x="Model", y="Delta", hue="Metric")
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.title("Effect of Adding Shortcut-Risk Metadata to Feature Models")
    plt.xlabel("")
    plt.ylabel("Metric change vs ECG-only")
    plt.xticks(rotation=25, ha="right")
    save_plot("validation_metadata_ablation_delta")
else:
    metadataAblationResultsDf = pd.DataFrame()
    print("No metadata shortcut-risk columns were available, so the ablation was skipped.")

def buildFreshFeatureModel(modelName, yTrainLocal):
    model = clone(classicalModels[modelName])
    if modelName == "XGBoost Features":
        scalePosWeightLocal = (len(yTrainLocal) - yTrainLocal.sum()) / max(yTrainLocal.sum(), 1)
        model.set_params(scale_pos_weight=scalePosWeightLocal)
    return model

trainValPatientDf = splitDf[splitDf["split"].isin(["train", "validation"])][["patient_id", "target"]].reset_index(drop=True)
accessibleStabilityModels = list(classicalModels.keys()) + ["Echo State Network (ESN)"]
stabilityCandidateNames = clinicalResultsDf[
    clinicalResultsDf["Model"].isin(accessibleStabilityModels)
]["Model"].head(3).tolist()

stabilityRecords = []
stabilityNote = (
    "Deep end-to-end and transfer models are not included in this repeated split check to keep runtime manageable. "
    "The stability analysis therefore focuses on interpretable or fast ECG-only finalists."
)
print(stabilityNote)
print("Stability candidates:", stabilityCandidateNames)

outerSplitter = StratifiedShuffleSplit(n_splits=6, test_size=0.25, random_state=randomState)

for splitNumber, (trainPoolIdx, holdoutIdx) in enumerate(
    outerSplitter.split(trainValPatientDf["patient_id"], trainValPatientDf["target"]),
    start=1,
):
    trainPoolDf = trainValPatientDf.iloc[trainPoolIdx].reset_index(drop=True)
    holdoutDf = trainValPatientDf.iloc[holdoutIdx].reset_index(drop=True)

    innerSplitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=randomState + splitNumber)
    innerTrainIdx, tuneIdx = next(innerSplitter.split(trainPoolDf["patient_id"], trainPoolDf["target"]))
    innerTrainDf = trainPoolDf.iloc[innerTrainIdx].reset_index(drop=True)
    tuneDf = trainPoolDf.iloc[tuneIdx].reset_index(drop=True)

    innerTrainIds = innerTrainDf["patient_id"].tolist()
    tuneIds = tuneDf["patient_id"].tolist()
    holdoutIds = holdoutDf["patient_id"].tolist()

    for modelName in stabilityCandidateNames:
        if modelName in classicalModels:
            xTrainLocalDf, yTrainLocal = getTabularSplit(innerTrainIds, includeMetadata=False)
            xTuneLocalDf, yTuneLocal = getTabularSplit(tuneIds, includeMetadata=False)
            xHoldoutLocalDf, yHoldoutLocal = getTabularSplit(holdoutIds, includeMetadata=False)

            localImputer = SimpleImputer(strategy="median")
            xTrainLocal = localImputer.fit_transform(xTrainLocalDf)
            xTuneLocal = localImputer.transform(xTuneLocalDf)
            xHoldoutLocal = localImputer.transform(xHoldoutLocalDf)

            localScaler = StandardScaler()
            xTrainLocalScaled = localScaler.fit_transform(xTrainLocal)
            xTuneLocalScaled = localScaler.transform(xTuneLocal)
            xHoldoutLocalScaled = localScaler.transform(xHoldoutLocal)

            if "SVM" in modelName:
                xTrainInput, xTuneInput, xHoldoutInput = xTrainLocalScaled, xTuneLocalScaled, xHoldoutLocalScaled
            else:
                xTrainInput, xTuneInput, xHoldoutInput = xTrainLocal, xTuneLocal, xHoldoutLocal

            localModel = buildFreshFeatureModel(modelName, yTrainLocal)
            if modelName == "AdaBoost":
                localWeights = makeSampleWeightVector(yTrainLocal)
                localModel.fit(xTrainInput, yTrainLocal, sample_weight=localWeights)
            else:
                localModel.fit(xTrainInput, yTrainLocal)

            tuneProba = localModel.predict_proba(xTuneInput)[:, 1].astype(float)
            bestThreshold, _ = tuneThreshold(yTuneLocal, tuneProba)
            holdoutProba = localModel.predict_proba(xHoldoutInput)[:, 1].astype(float)
            metricRow, _ = calculateBinaryMetrics(yHoldoutLocal, holdoutProba, bestThreshold)
        else:
            xTrainLocal, yTrainLocal = getSequenceSplit(innerTrainIds)
            xTuneLocal, yTuneLocal = getSequenceSplit(tuneIds)
            xHoldoutLocal, yHoldoutLocal = getSequenceSplit(holdoutIds)

            xTrainReservoir = esnTransform(xTrainLocal, seed=randomState + splitNumber)
            xTuneReservoir = esnTransform(xTuneLocal, seed=randomState + splitNumber)
            xHoldoutReservoir = esnTransform(xHoldoutLocal, seed=randomState + splitNumber)

            localScaler = StandardScaler()
            xTrainReservoir = localScaler.fit_transform(xTrainReservoir)
            xTuneReservoir = localScaler.transform(xTuneReservoir)
            xHoldoutReservoir = localScaler.transform(xHoldoutReservoir)

            localClassifier = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=randomState + splitNumber,
            )
            localClassifier.fit(xTrainReservoir, yTrainLocal)
            tuneProba = localClassifier.predict_proba(xTuneReservoir)[:, 1].astype(float)
            bestThreshold, _ = tuneThreshold(yTuneLocal, tuneProba)
            holdoutProba = localClassifier.predict_proba(xHoldoutReservoir)[:, 1].astype(float)
            metricRow, _ = calculateBinaryMetrics(yHoldoutLocal, holdoutProba, bestThreshold)

        metricRow.update({"Model": modelName, "Split": splitNumber})
        stabilityRecords.append(metricRow)

stabilityDetailDf = pd.DataFrame(stabilityRecords)
stabilityDetailFile = predictionDir / make_filename("stability repeated split details", "csv")
stabilitySummaryFile = predictionDir / make_filename("stability repeated split summary", "csv")
stabilityDetailDf.to_csv(stabilityDetailFile, index=False)

stabilitySummaryDf = stabilityDetailDf.groupby("Model").agg(
    {
        "AUC": ["mean", "std"],
        "PR AUC": ["mean", "std"],
        "F1": ["mean", "std"],
        "Sensitivity": ["mean", "std"],
        "Specificity": ["mean", "std"],
        "Balanced Accuracy": ["mean", "std"],
        "False Negatives": ["mean", "std"],
    }
)
stabilitySummaryDf.columns = [f"{metric} {stat}" for metric, stat in stabilitySummaryDf.columns]
stabilitySummaryDf = stabilitySummaryDf.reset_index().sort_values(
    by=["Sensitivity mean", "PR AUC mean", "F1 mean"],
    ascending=[False, False, False],
)
stabilitySummaryDf.to_csv(stabilitySummaryFile, index=False)

print("Saved repeated split detail file:", repoDisplayPath(stabilityDetailFile))
print("Saved repeated split summary file:", repoDisplayPath(stabilitySummaryFile))
display(stabilitySummaryDf)

plt.figure(figsize=(10, 5))
sns.boxplot(data=stabilityDetailDf, x="Model", y="Sensitivity")
plt.title("Repeated Split Sensitivity Stability on the Train+Validation Pool")
plt.xlabel("")
plt.ylabel("Sensitivity")
plt.xticks(rotation=25, ha="right")
save_plot("validation_repeated_split_sensitivity_boxplot")

def minMaxScore(series, higherIsBetter=True, neutralValue=0.5):
    values = pd.to_numeric(pd.Series(series), errors="coerce").astype(float)
    validValues = values.dropna()

    if validValues.empty or np.isclose(validValues.max(), validValues.min()):
        return pd.Series(neutralValue, index=values.index, dtype=float)

    score = (values - validValues.min()) / (validValues.max() - validValues.min())
    if not higherIsBetter:
        score = 1.0 - score
    return score.fillna(neutralValue).clip(0.0, 1.0)

def computeParetoFlags(df, maximizeColumns, minimizeColumns=None):
    minimizeColumns = minimizeColumns or []
    paretoFlags = []

    for rowIdx, row in df.iterrows():
        dominated = False
        for otherIdx, otherRow in df.iterrows():
            if rowIdx == otherIdx:
                continue

            betterOrEqual = all(otherRow[column] >= row[column] for column in maximizeColumns)
            betterOrEqual = betterOrEqual and all(otherRow[column] <= row[column] for column in minimizeColumns)
            strictlyBetter = any(otherRow[column] > row[column] for column in maximizeColumns)
            strictlyBetter = strictlyBetter or any(otherRow[column] < row[column] for column in minimizeColumns)

            if betterOrEqual and strictlyBetter:
                dominated = True
                break

        paretoFlags.append(not dominated)

    return paretoFlags

multiAspectRankingDf = benchmarkResultsDf.copy()
multiAspectRankingDf["Training Effort"] = (
    multiAspectRankingDf["Epochs Trained"].fillna(1.0)
    + multiAspectRankingDf["Pretrain Epochs"].fillna(0.0)
).clip(lower=1.0)
multiAspectRankingDf["Validation F1 Gap"] = (
    multiAspectRankingDf["F1"] - multiAspectRankingDf["Validation F1"]
).abs()
multiAspectRankingDf["Consistency Score"] = (1.0 - multiAspectRankingDf["Validation F1 Gap"]).clip(0.0, 1.0)
multiAspectRankingDf["Performance Score"] = (
    0.45 * multiAspectRankingDf["PR AUC"]
    + 0.30 * multiAspectRankingDf["F1"]
    + 0.25 * multiAspectRankingDf["Balanced Accuracy"]
)
multiAspectRankingDf["Safety Score"] = (
    0.50 * multiAspectRankingDf["Sensitivity"]
    + 0.30 * multiAspectRankingDf["Specificity"]
    + 0.20 * multiAspectRankingDf["Precision"]
)
multiAspectRankingDf["Efficiency Score"] = minMaxScore(np.log1p(multiAspectRankingDf["Training Effort"]), higherIsBetter=False)

stabilityMergeDf = stabilitySummaryDf[
    ["Model", "PR AUC mean", "F1 mean", "Sensitivity mean", "Sensitivity std", "False Negatives std"]
].copy()
multiAspectRankingDf = multiAspectRankingDf.merge(stabilityMergeDf, on="Model", how="left")
multiAspectRankingDf["Stability Performance Score"] = (
    0.40 * minMaxScore(multiAspectRankingDf["PR AUC mean"])
    + 0.20 * minMaxScore(multiAspectRankingDf["F1 mean"])
    + 0.40 * minMaxScore(multiAspectRankingDf["Sensitivity mean"])
)
multiAspectRankingDf["Stability Variance Score"] = (
    0.60 * minMaxScore(multiAspectRankingDf["Sensitivity std"], higherIsBetter=False)
    + 0.40 * minMaxScore(multiAspectRankingDf["False Negatives std"], higherIsBetter=False)
)
multiAspectRankingDf["Stability Score"] = (
    0.5 * multiAspectRankingDf["Stability Performance Score"]
    + 0.5 * multiAspectRankingDf["Stability Variance Score"]
)
multiAspectRankingDf["Stability Evidence"] = np.where(
    multiAspectRankingDf["PR AUC mean"].notna(),
    "Repeated split available",
    "Not run; neutral fill",
)
multiAspectRankingDf["Balanced Composite Score"] = (
    0.40 * multiAspectRankingDf["Performance Score"]
    + 0.35 * multiAspectRankingDf["Safety Score"]
    + 0.15 * multiAspectRankingDf["Consistency Score"]
    + 0.05 * multiAspectRankingDf["Efficiency Score"]
    + 0.05 * multiAspectRankingDf["Stability Score"]
)
multiAspectRankingDf["Performance/Safety Pareto"] = computeParetoFlags(
    multiAspectRankingDf,
    maximizeColumns=["PR AUC", "F1", "Sensitivity", "Specificity", "Balanced Accuracy"],
    minimizeColumns=["False Negatives"],
)
multiAspectRankingDf = multiAspectRankingDf.sort_values(
    by=[
        "Balanced Composite Score",
        "Performance/Safety Pareto",
        "Safety Score",
        "Performance Score",
        "Consistency Score",
        "Efficiency Score",
    ],
    ascending=[False, False, False, False, False, False],
).reset_index(drop=True)
multiAspectRankingDf.insert(0, "Balanced Rank", np.arange(1, len(multiAspectRankingDf) + 1))

multiAspectRankingFile = outputRoot / make_filename("multi aspect ranking", "csv")
paretoFrontFile = outputRoot / make_filename("pareto front models", "csv")
multiAspectRankingDf.to_csv(multiAspectRankingFile, index=False)
multiAspectRankingDf[multiAspectRankingDf["Performance/Safety Pareto"]].to_csv(paretoFrontFile, index=False)

print(
    "Balanced composite weights: "
    "performance 0.40, safety 0.35, consistency 0.15, efficiency 0.05, stability 0.05."
)
print("Saved multi-aspect ranking:", repoDisplayPath(multiAspectRankingFile))
print("Saved Pareto-front model list:", repoDisplayPath(paretoFrontFile))
display(
    multiAspectRankingDf[
        [
            "Balanced Rank",
            "Model",
            "Balanced Composite Score",
            "Performance Score",
            "Safety Score",
            "Consistency Score",
            "Efficiency Score",
            "Stability Score",
            "Performance/Safety Pareto",
            "Stability Evidence",
        ]
    ]
)

featureInterpretationRows = benchmarkResultsDf[
    benchmarkResultsDf["Model"].isin(list(trainedClassicalModels.keys()))
].sort_values(by=["Sensitivity", "PR AUC", "F1"], ascending=[False, False, False])

if not featureInterpretationRows.empty:
    featureInterpretationModel = featureInterpretationRows.iloc[0]["Model"]
    fittedFeatureModel = trainedClassicalModels[featureInterpretationModel]
    featureInput = xTestTabScaled if "SVM" in featureInterpretationModel else xTestTab

    permutationResult = permutation_importance(
        fittedFeatureModel,
        featureInput,
        yTestTab,
        scoring="average_precision",
        n_repeats=20,
        random_state=randomState,
    )
    featureImportanceDf = pd.DataFrame(
        {
            "Feature": xTestTabDf.columns,
            "Importance Mean": permutationResult.importances_mean,
            "Importance Std": permutationResult.importances_std,
        }
    ).sort_values("Importance Mean", ascending=False)
    featureImportanceDf["Lead Group"] = (
        featureImportanceDf["Feature"]
        .str.extract(r"^(I|II|III|aVR|aVL|aVF|V1|V2|V3|V4|V5|V6)")[0]
        .fillna("cross-lead / global")
    )
    featureImportanceFile = predictionDir / make_filename("feature permutation importance", "csv")
    featureImportanceDf.to_csv(featureImportanceFile, index=False)

    leadImportanceDf = featureImportanceDf.groupby("Lead Group", as_index=False)["Importance Mean"].sum().sort_values(
        "Importance Mean",
        ascending=False,
    )
    leadImportanceFile = predictionDir / make_filename("feature permutation importance by lead", "csv")
    leadImportanceDf.to_csv(leadImportanceFile, index=False)

    print("Feature interpretability model:", featureInterpretationModel)
    print("Saved permutation importance:", repoDisplayPath(featureImportanceFile))
    print("Saved lead-group importance:", repoDisplayPath(leadImportanceFile))
    display(featureImportanceDf.head(15))
    display(leadImportanceDf)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=featureImportanceDf.head(15), x="Importance Mean", y="Feature", color="#2a9d8f")
    plt.title(f"Top Permutation Importance Features: {featureInterpretationModel}")
    plt.xlabel("Average PR-AUC importance")
    plt.ylabel("")
    save_plot("interpretability_feature_permutation_importance")

    plt.figure(figsize=(8, 4))
    sns.barplot(data=leadImportanceDf, x="Lead Group", y="Importance Mean", color="#e76f51")
    plt.title("Lead-Level Importance Aggregation for the Feature Model")
    plt.xlabel("")
    plt.ylabel("Summed feature importance")
    plt.xticks(rotation=25, ha="right")
    save_plot("interpretability_feature_importance_by_lead")
else:
    print("No trained feature model was available for permutation importance.")

def loadSavedSequenceModel(modelName):
    artifactPath = trainedSequenceModelArtifacts.get(modelName)
    resolvedArtifactPath = repoResolvePath(artifactPath) if artifactPath else None
    if resolvedArtifactPath is None or not resolvedArtifactPath.exists():
        return None
    return tf.keras.models.load_model(
        resolvedArtifactPath,
        custom_objects={"LearnablePositionEmbedding": LearnablePositionEmbedding},
        compile=False,
    )

def computeLeadSaliency(model, xBatch):
    xTensor = tf.convert_to_tensor(np.asarray(xBatch).astype(np.float32))
    with tf.GradientTape() as tape:
        tape.watch(xTensor)
        outputs = model(xTensor, training=False)
    gradients = tape.gradient(outputs, xTensor)
    saliency = np.abs(gradients.numpy())
    return saliency.mean(axis=(0, 1)), saliency.mean(axis=0)

sequenceInterpretationRows = clinicalResultsDf[
    clinicalResultsDf["Model"].isin(list(trainedSequenceModelArtifacts.keys()))
].copy()
if not sequenceInterpretationRows.empty:
    sequenceInterpretationModel = sequenceInterpretationRows.iloc[0]["Model"]
    sequencePredictionDf = benchmarkPredictionTables[sequenceInterpretationModel]
    positiveSequenceDf = sequencePredictionDf[sequencePredictionDf["target"] == 1].sort_values("pred_proba", ascending=False)

    if not positiveSequenceDf.empty:
        selectedPatientIds = positiveSequenceDf["patient_id"].astype(int).head(5).tolist()
        selectedIndices = [testIds.index(pid) for pid in selectedPatientIds if pid in testIds]
        sequenceModel = loadSavedSequenceModel(sequenceInterpretationModel)

        if sequenceModel is not None and selectedIndices:
            leadSaliency, timeLeadSaliency = computeLeadSaliency(sequenceModel, xTestSeq[selectedIndices])
            sequenceLeadSaliencyDf = pd.DataFrame({"Lead": standardLeads, "Saliency": leadSaliency}).sort_values(
                "Saliency",
                ascending=False,
            )
            sequenceLeadSaliencyFile = predictionDir / make_filename("sequence lead saliency", "csv")
            sequenceLeadSaliencyDf.to_csv(sequenceLeadSaliencyFile, index=False)

            print("Sequence saliency model:", sequenceInterpretationModel)
            print("Saliency patients:", selectedPatientIds)
            print("Saved sequence lead saliency:", repoDisplayPath(sequenceLeadSaliencyFile))
            display(sequenceLeadSaliencyDf)

            plt.figure(figsize=(8, 4))
            sns.barplot(data=sequenceLeadSaliencyDf, x="Lead", y="Saliency", color="#264653")
            plt.title(f"Lead Saliency for the Sequence Model: {sequenceInterpretationModel}")
            plt.xlabel("")
            plt.ylabel("Mean |gradient|")
            plt.xticks(rotation=25, ha="right")
            save_plot("interpretability_sequence_lead_saliency")

            v123Indices = [standardLeads.index(leadName) for leadName in ["V1", "V2", "V3"]]
            plt.figure(figsize=(12, 3))
            sns.heatmap(
                timeLeadSaliency[:, v123Indices].T,
                cmap="magma",
                cbar_kws={"label": "Mean |gradient|"},
            )
            plt.yticks(np.arange(len(v123Indices)) + 0.5, ["V1", "V2", "V3"], rotation=0)
            plt.xlabel("Median beat sample")
            plt.ylabel("Lead")
            plt.title(f"Saliency Heatmap on V1-V3: {sequenceInterpretationModel}")
            save_plot("interpretability_sequence_v1_v3_saliency_heatmap")

            del sequenceModel
            tf.keras.backend.clear_session()
        else:
            print("Sequence saliency was skipped because the saved model artifact could not be loaded.")
    else:
        print("Sequence saliency was skipped because no Brugada-positive test cases were available.")
else:
    print("No saved sequence model was available for saliency analysis.")

clinicalRow = clinicalResultsDf.iloc[0]
clinicalModelName = clinicalRow["Model"]
clinicalPredictionDf = benchmarkPredictionTables[clinicalModelName].copy()

thresholdScanDf = evaluateThresholdGrid(clinicalPredictionDf["target"], clinicalPredictionDf["pred_proba"])
thresholdScanFile = predictionDir / make_filename("clinical candidate threshold scan", "csv")
thresholdScanDf.to_csv(thresholdScanFile, index=False)

plt.figure(figsize=(10, 5))
plt.plot(thresholdScanDf["Threshold"], thresholdScanDf["Sensitivity"], label="Sensitivity")
plt.plot(thresholdScanDf["Threshold"], thresholdScanDf["Specificity"], label="Specificity")
plt.plot(thresholdScanDf["Threshold"], thresholdScanDf["F1"], label="F1")
plt.axvline(float(clinicalRow["Threshold"]), color="black", linestyle="--", label="Selected threshold")
plt.title(f"Threshold Trade-off for the Clinical Candidate: {clinicalModelName}")
plt.xlabel("Threshold")
plt.ylabel("Metric value")
plt.legend()
save_plot("failure_analysis_clinical_candidate_threshold_tradeoff")

analysisColumns = [column for column in ["patient_id", "basal_pattern", "sudden_death"] if column in metadata.columns]
if analysisColumns:
    clinicalFailureDf = clinicalPredictionDf.merge(metadata[analysisColumns].copy(), on="patient_id", how="left")
else:
    clinicalFailureDf = clinicalPredictionDf.copy()

clinicalFailureDf["Outcome Type"] = np.select(
    [
        (clinicalFailureDf["target"] == 1) & (clinicalFailureDf["pred_label"] == 1),
        (clinicalFailureDf["target"] == 1) & (clinicalFailureDf["pred_label"] == 0),
        (clinicalFailureDf["target"] == 0) & (clinicalFailureDf["pred_label"] == 1),
        (clinicalFailureDf["target"] == 0) & (clinicalFailureDf["pred_label"] == 0),
    ],
    ["True Positive", "False Negative", "False Positive", "True Negative"],
    default="Other",
)
clinicalFailureFile = predictionDir / make_filename("clinical candidate failure overview", "csv")
clinicalFailureDf.to_csv(clinicalFailureFile, index=False)

falseNegativeDf = clinicalFailureDf[clinicalFailureDf["Outcome Type"] == "False Negative"].sort_values("pred_proba")
falseNegativeFile = predictionDir / make_filename("clinical candidate false negatives", "csv")
falseNegativeDf.to_csv(falseNegativeFile, index=False)

print("Saved threshold scan:", repoDisplayPath(thresholdScanFile))
print("Saved failure overview:", repoDisplayPath(clinicalFailureFile))
print("Saved false negative table:", repoDisplayPath(falseNegativeFile))
display(falseNegativeDf)

if "basal_pattern" in clinicalFailureDf.columns and not clinicalFailureDf[clinicalFailureDf["target"] == 1].empty:
    positiveOutcomeDf = clinicalFailureDf[clinicalFailureDf["target"] == 1].copy()
    positiveOutcomeDf["basal_pattern_display"] = positiveOutcomeDf["basal_pattern"].fillna("missing")
    plt.figure(figsize=(8, 4))
    sns.countplot(data=positiveOutcomeDf, x="Outcome Type", hue="basal_pattern_display")
    plt.title("Brugada Positive Outcomes by Basal Pattern")
    plt.xlabel("")
    plt.ylabel("Count")
    save_plot("failure_analysis_basal_pattern_vs_outcome")

for patientId in falseNegativeDf["patient_id"].astype(int).head(2).tolist():
    if int(patientId) not in sequenceCache:
        continue
    signal = sequenceCache[int(patientId)]
    timeAxisMs = np.arange(signal.shape[0]) * 1000.0 / fsExpected
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for ax, leadName in zip(axes, ["V1", "V2", "V3"]):
        leadIdx = standardLeads.index(leadName)
        ax.plot(timeAxisMs, signal[:, leadIdx], color="black", linewidth=1.2)
        ax.set_ylabel(leadName)
    axes[0].set_title(f"False Negative Brugada Example: patient_id={patientId}")
    axes[-1].set_xlabel("Time (ms)")
    plt.tight_layout()
    save_plot(f"failure_false_negative_patient_{patientId}_v1_v3")
