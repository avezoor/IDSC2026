targetColumn = "brugada"
assert targetColumn in metadata.columns, f"Expected '{targetColumn}' in metadata columns."

labelCounts = metadata[targetColumn].value_counts(dropna=False).sort_index()
positiveCount = int((metadata[targetColumn] == 1).sum())
negativeCount = int((metadata[targetColumn] == 0).sum())
imbalanceRatio = negativeCount / max(positiveCount, 1)
positiveRate = positiveCount / max(len(metadata), 1)

print(labelCounts)
print()
print(metadata[targetColumn].value_counts(normalize=True).rename("proportion"))
print()
print(f"Minority class (Brugada) count : {positiveCount}")
print(f"Majority class (Normal) count  : {negativeCount}")
print(f"Positive rate                  : {positiveRate:.4f}")
print(f"Imbalance ratio (Normal/Brugada): {imbalanceRatio:.2f}")
print(
    "Imbalance handling plan: stratified patient split, class weighting or scale_pos_weight, "
    "threshold tuning on validation data, and model ranking that emphasizes PR-AUC, F1, sensitivity, "
    "and Brugada detection instead of raw accuracy alone."
)

plt.figure(figsize=(5, 4))
sns.countplot(x=metadata[targetColumn])
plt.title("Label Distribution: 0 Normal, 1 Brugada")
plt.xlabel("Target")
plt.ylabel("Count")
save_plot("eda_label_distribution")

for column_name in ["basal_pattern", "sudden_death"]:
    if column_name in metadata.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=metadata[column_name].fillna("missing"))
        plt.title(f"Metadata Distribution: {column_name}")
        plt.xlabel(column_name)
        plt.ylabel("Count")
        plt.xticks(rotation=20)
        save_plot(f"eda_{column_name}_distribution")

heaFiles = list(datasetRoot.rglob("*.hea"))
print("Total .hea files found:", len(heaFiles))

recordMap = {}
for hea in heaFiles:
    patient_key = Path(hea).stem
    recordMap[str(patient_key)] = str(Path(hea).with_suffix(""))

print("Indexed records:", len(recordMap))

missing_ids = [
    str(patient_id)
    for patient_id in metadata["patient_id"].astype(str)
    if str(patient_id) not in recordMap
]
print("Missing patient IDs in file index:", len(missing_ids))
if missing_ids:
    print("Example missing IDs:", missing_ids[:10])

expectedSamplingRate = 100
expectedDurationSeconds = 12
expectedSignalLength = expectedSamplingRate * expectedDurationSeconds

headerSummary = []
for patient_id, record_path in tqdm(sorted(recordMap.items()), total=len(recordMap), desc="Checking WFDB headers"):
    header = wfdb.rdheader(record_path)
    headerSummary.append(
        {
            "patient_id": patient_id,
            "fs": float(header.fs),
            "sig_len": int(header.sig_len),
            "n_sig": int(header.n_sig),
        }
    )

headerDf = pd.DataFrame(headerSummary)
assert not headerDf.empty, "No WFDB headers were indexed from the dataset root."

headerChecks = [
    ("Sampling rate", bool((headerDf["fs"] == expectedSamplingRate).all()), f"all records should be {expectedSamplingRate} Hz"),
    ("Signal length", bool((headerDf["sig_len"] == expectedSignalLength).all()), f"all records should be {expectedSignalLength} samples ({expectedDurationSeconds} seconds)"),
    ("Lead count", bool((headerDf["n_sig"] == len(standardLeads)).all()), f"all records should contain {len(standardLeads)} leads"),
]
headerIssues = []
for label, passed, expectation in headerChecks:
    status = "OK" if passed else "WARNING"
    print(f"{label} check: {status} ({expectation})")
    if not passed:
        headerIssues.append(f"{label} mismatch: {expectation}.")

if headerIssues:
    print("Recording specification cross-check warnings:")
    for issue in headerIssues:
        print("-", issue)
    print("Execution will continue with the detected recording properties.")
else:
    print("All records passed the recording specification cross-check.")

print("Expected sampling rate:", expectedSamplingRate, "Hz")
print("Expected signal length:", expectedSignalLength, "samples")
display(headerDf.head())

def standardizeLeadNames(sigNames):
    mapping = {}
    for name in sigNames:
        normalized = (
            str(name)
            .strip()
            .replace("AVR", "aVR")
            .replace("AVL", "aVL")
            .replace("AVF", "aVF")
        )
        mapping[name] = normalized
    return mapping

def loadEcgRecord(patientId):
    patientId = str(patientId)
    assert patientId in recordMap, f"Record for patient_id {patientId} not found."

    record = wfdb.rdrecord(recordMap[patientId])
    signal_df = pd.DataFrame(record.p_signal, columns=record.sig_name)
    signal_df = signal_df.rename(columns=standardizeLeadNames(signal_df.columns.tolist()))

    for lead_name in standardLeads:
        if lead_name not in signal_df.columns:
            raise ValueError(f"Lead {lead_name} not found for patient_id {patientId}")

    signal_df = signal_df[standardLeads].copy()
    return signal_df.values.astype(np.float32), standardLeads, int(record.fs)

normalId = metadata.loc[metadata[targetColumn] == 0, "patient_id"].iloc[0]
brugadaId = metadata.loc[metadata[targetColumn] == 1, "patient_id"].iloc[0]

def plotSelectedLeads(patientId, leadsToPlot=("V1", "V2", "V3", "II"), titlePrefix=""):
    X, leads, fs = loadEcgRecord(patientId)
    t = np.arange(len(X)) / fs

    plt.figure(figsize=(14, 8))
    for i, lead_name in enumerate(leadsToPlot, 1):
        lead_index = leads.index(lead_name)
        plt.subplot(len(leadsToPlot), 1, i)
        plt.plot(t, X[:, lead_index], linewidth=1.0)
        plt.title(f"{titlePrefix} patient_id={patientId} lead={lead_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("mV")
    plt.tight_layout()
    save_plot(f"eda_{titlePrefix.lower()}_{patientId}_selected_leads")

plotSelectedLeads(normalId, titlePrefix="NORMAL")
plotSelectedLeads(brugadaId, titlePrefix="BRUGADA")
