benchmarkDf = featuresDf.copy()
requiredBenchmarkColumns = {"patient_id", "target"}
missingBenchmarkColumns = requiredBenchmarkColumns.difference(benchmarkDf.columns)
assert not missingBenchmarkColumns, (
    "The feature table is missing required columns "
    f"{sorted(missingBenchmarkColumns)}. Check preprocessing.py for earlier failures."
)
benchmarkDf["patient_id"] = pd.to_numeric(benchmarkDf["patient_id"], errors="coerce")
benchmarkDf["target"] = pd.to_numeric(benchmarkDf["target"], errors="coerce")
benchmarkDf = benchmarkDf.dropna(subset=["patient_id", "target"]).copy()
benchmarkDf = benchmarkDf[benchmarkDf["target"].isin([0, 1])].copy()
benchmarkDf["patient_id"] = benchmarkDf["patient_id"].astype(int)
benchmarkDf["target"] = benchmarkDf["target"].astype(int)
benchmarkDf = benchmarkDf.drop_duplicates(subset=["patient_id"]).reset_index(drop=True)

assert not benchmarkDf.empty, (
    "The feature table is empty after preprocessing and label filtering. "
    "Check the preprocessing log for skipped patients."
)

sequenceCache = {}
validPatients = []
for row in tqdm(benchmarkDf[["patient_id", "target"]].itertuples(index=False), total=len(benchmarkDf)):
    pid = int(row.patient_id)
    try:
        sequenceCache[pid] = buildSequenceSample(pid, focusLeads=standardLeads)
        validPatients.append({"patient_id": pid, "target": int(row.target)})
    except Exception as exc:
        print(f"Skipping patient_id={pid} because median beat extraction failed: {exc}")

benchmarkPatientDf = pd.DataFrame(validPatients).sort_values("patient_id").reset_index(drop=True)
benchmarkDf = benchmarkDf[benchmarkDf["patient_id"].isin(benchmarkPatientDf["patient_id"])].copy()
benchmarkDf = benchmarkDf.sort_values("patient_id").reset_index(drop=True)

trainDf, testDf = train_test_split(
    benchmarkPatientDf,
    test_size=0.20,
    stratify=benchmarkPatientDf["target"],
    random_state=randomState,
)
trainDf, valDf = train_test_split(
    trainDf,
    test_size=0.20,
    stratify=trainDf["target"],
    random_state=randomState,
)

splitDf = pd.concat(
    [
        trainDf.assign(split="train"),
        valDf.assign(split="validation"),
        testDf.assign(split="test"),
    ],
    axis=0,
).reset_index(drop=True)
splitFile = predictionDir / make_filename("shared patient split", "csv")
splitDf.to_csv(splitFile, index=False)

trainIds = trainDf["patient_id"].tolist()
valIds = valDf["patient_id"].tolist()
testIds = testDf["patient_id"].tolist()

labelByPatient = benchmarkPatientDf.set_index("patient_id")["target"].to_dict()

metadataShortcutColumns = [column for column in ["basal_pattern_meta", "sudden_death_meta"] if column in benchmarkDf.columns]
primaryExcludedColumns = ["patient_id", "target"] + metadataShortcutColumns
featureColumns = [column for column in benchmarkDf.columns if column not in primaryExcludedColumns]
featureColumnsWithMetadata = [column for column in benchmarkDf.columns if column not in ["patient_id", "target"]]

featureByPatient = benchmarkDf.set_index("patient_id")[featureColumns].sort_index()
featureByPatientWithMetadata = benchmarkDf.set_index("patient_id")[featureColumnsWithMetadata].sort_index()

def getTabularSplit(patientIds, includeMetadata=False):
    featureFrame = featureByPatientWithMetadata if includeMetadata else featureByPatient
    x_df = featureFrame.loc[patientIds].copy()
    y = np.array([labelByPatient[int(pid)] for pid in patientIds], dtype=int)
    return x_df, y

def getSequenceSplit(patientIds):
    X = np.stack([sequenceCache[int(pid)] for pid in patientIds]).astype(np.float32)
    y = np.array([labelByPatient[int(pid)] for pid in patientIds], dtype=int)
    return X, y

xTrainTabDf, yTrainTab = getTabularSplit(trainIds, includeMetadata=False)
xValTabDf, yValTab = getTabularSplit(valIds, includeMetadata=False)
xTestTabDf, yTestTab = getTabularSplit(testIds, includeMetadata=False)

benchmarkImputer = SimpleImputer(strategy="median")
xTrainTab = benchmarkImputer.fit_transform(xTrainTabDf)
xValTab = benchmarkImputer.transform(xValTabDf)
xTestTab = benchmarkImputer.transform(xTestTabDf)

benchmarkScaler = StandardScaler()
xTrainTabScaled = benchmarkScaler.fit_transform(xTrainTab)
xValTabScaled = benchmarkScaler.transform(xValTab)
xTestTabScaled = benchmarkScaler.transform(xTestTab)

xTrainSeq, yTrainSeq = getSequenceSplit(trainIds)
xValSeq, yValSeq = getSequenceSplit(valIds)
xTestSeq, yTestSeq = getSequenceSplit(testIds)

featureSetAuditDf = pd.DataFrame(
    [
        {
            "Feature Set": "ECG-only primary benchmark",
            "Feature Count": int(len(featureColumns)),
            "Shortcut-risk columns included": "No",
            "Excluded columns": ", ".join(metadataShortcutColumns) if metadataShortcutColumns else "None",
        },
        {
            "Feature Set": "ECG + metadata ablation",
            "Feature Count": int(len(featureColumnsWithMetadata)),
            "Shortcut-risk columns included": "Yes" if metadataShortcutColumns else "No",
            "Excluded columns": "patient_id, target",
        },
    ]
)
featureAuditFile = predictionDir / make_filename("feature set audit", "csv")
featureSetAuditDf.to_csv(featureAuditFile, index=False)

print("Shared split file:", repoDisplayPath(splitFile))
print("Feature audit file:", repoDisplayPath(featureAuditFile))
print("Benchmark patient count:", len(benchmarkPatientDf))
print("Train / Validation / Test:", len(trainIds), len(valIds), len(testIds))
print("Train labels:", np.unique(yTrainTab, return_counts=True))
print("Validation labels:", np.unique(yValTab, return_counts=True))
print("Test labels:", np.unique(yTestTab, return_counts=True))
print("ECG-only feature count:", len(featureColumns))
print("Metadata shortcut-risk columns excluded from the primary benchmark:", metadataShortcutColumns if metadataShortcutColumns else "None")
print("Tabular train shape:", xTrainTab.shape)
print("Sequence train shape:", xTrainSeq.shape)
display(featureSetAuditDf)

plt.figure(figsize=(6, 4))
sns.countplot(data=splitDf, x="split", hue="target")
plt.title("Shared Patient Split by Class")
plt.xlabel("Split")
plt.ylabel("Patient count")
save_plot("test_split_shared_patient_distribution")

display(splitDf.head())
