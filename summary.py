leaderboardRow = benchmarkResultsDf.iloc[0]
clinicalRow = clinicalResultsDf.iloc[0]
sameModel = leaderboardRow["Model"] == clinicalRow["Model"]
sensitivityGap = float(clinicalRow["Sensitivity"] - leaderboardRow["Sensitivity"])
falseNegativeGap = int(leaderboardRow["False Negatives"] - clinicalRow["False Negatives"])
falsePositiveGap = int(clinicalRow["False Positives"] - leaderboardRow["False Positives"])

print("Primary benchmark note:")
print("ECG-only is the main benchmark. Shortcut-risk metadata columns are excluded from model training and reported only as a separate ablation.")
print()

print("Leaderboard winner:")
print(
    f"{leaderboardRow['Model']} | PR-AUC={leaderboardRow['PR AUC']:.4f} | F1={leaderboardRow['F1']:.4f} | "
    f"Sensitivity={leaderboardRow['Sensitivity']:.4f} | False Negatives={int(leaderboardRow['False Negatives'])}"
)
print()

print("Safety-oriented clinical candidate:")
print(
    f"{clinicalRow['Model']} | Sensitivity={clinicalRow['Sensitivity']:.4f} | "
    f"False Negatives={int(clinicalRow['False Negatives'])} | PR-AUC={clinicalRow['PR AUC']:.4f} | "
    f"F1={clinicalRow['F1']:.4f}"
)
print()

if sameModel:
    print("Decision note:")
    print("The same model leads both the leaderboard and the safety-oriented ranking on this split.")
else:
    print("Decision note:")
    print(
        "The leaderboard winner and the safety-oriented clinical candidate are different. "
        "For the submission narrative, report the leaderboard winner as the top aggregate scorer and explicitly discuss the clinical candidate when talking about false-negative risk."
    )
    if falseNegativeGap > 0:
        falseNegativeMessage = f"reduces false negatives by {falseNegativeGap} case(s)"
    elif falseNegativeGap < 0:
        falseNegativeMessage = f"increases false negatives by {abs(falseNegativeGap)} case(s)"
    else:
        falseNegativeMessage = "keeps the same false-negative count"
    print(
        f"Clinical-safety gap: the clinical candidate changes sensitivity by {sensitivityGap:+.4f}, {falseNegativeMessage}, "
        f"and changes false positives by {falsePositiveGap:+d} case(s)."
    )
    if sensitivityGap > 0 and falseNegativeGap > 0:
        print(
            "This is the core trade-off to discuss in the report: the aggregate winner may still be weaker for a safety-oriented Brugada story if it misses more true Brugada cases."
        )
print()

if "metadataAblationResultsDf" in globals() and not metadataAblationResultsDf.empty:
    print("Metadata ablation note:")
    print(
        "Feature-model results with ECG + metadata are saved separately so any apparent gain from shortcut-risk metadata can be discussed transparently instead of being mixed into the main benchmark."
    )
    print()

if "stabilitySummaryDf" in globals() and not stabilitySummaryDf.empty:
    stableTop = stabilitySummaryDf.iloc[0]
    print("Repeated split stability highlight:")
    print(
        f"{stableTop['Model']} | mean sensitivity={stableTop['Sensitivity mean']:.4f} +/- {stableTop['Sensitivity std']:.4f} | "
        f"mean PR-AUC={stableTop['PR AUC mean']:.4f}"
    )
    print(
        "This repeated split analysis improves validation rigor, but it does not fully remove the optimism risk introduced by comparing many models against one held-out test split."
    )
    print()

print("Responsible deployment note:")
print(
    "These results are research-only. Because the dataset is small, imbalanced, single-center, and evaluated in a multi-model selection setting, any deployment claim would require external validation, prospective threshold calibration, and a workflow that treats the model as decision support rather than autonomous diagnosis."
)
print()

print("Saved outputs:")
print("summary.csv:", summaryFile)
print("clinical_summary.csv:", clinicalSummaryFile)
print("shared_patient_split.csv:", splitFile)
print("feature_set_audit.csv:", featureAuditFile)
if "metadataAblationFile" in globals():
    print("metadata_ablation_summary.csv:", metadataAblationFile)
if "stabilitySummaryFile" in globals():
    print("stability_repeated_split_summary.csv:", stabilitySummaryFile)
print("prediction folder:", predictionDir)
print("plots folder:", plotDir)
print("logs.txt:", logFile)

if benchmarkFailures:
    print()
    print("Failed models during benchmarking:")
    for failureRow in benchmarkFailures:
        print(f"- {failureRow['Model']} | {failureRow['Error Type']}: {failureRow['Error Message']}")

print()
print("Log file content:")
if "_log_handle" in globals() and getattr(_log_handle, "closed", True) is False:
    _log_handle.flush()
with open(logFile, "r", encoding="utf-8") as fh:
    logText = fh.read()
_original_stdout.write(logText)
_original_stdout.flush()

display(benchmarkResultsDf)
display(clinicalResultsDf)
