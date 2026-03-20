def bandpassFilter(x, fs, low=0.5, high=40.0, order=3):
    nyquist = 0.5 * fs
    low = max(low / nyquist, 1e-4)
    high = min(high / nyquist, 0.99)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x, axis=0)

def robustScale(x):
    median = np.median(x, axis=0, keepdims=True)
    mad = np.median(np.abs(x - median), axis=0, keepdims=True)
    mad = np.where(mad < 1e-6, 1.0, mad)
    return (x - median) / (1.4826 * mad)

def detectRPeaks(signal1d, fs):
    signal = signal1d.copy()
    if len(signal) >= 11:
        signal = savgol_filter(signal, 11, 3)
    signal = signal - np.median(signal)

    prominence = max(0.15, np.std(signal) * 0.5)
    distance = int(0.45 * fs)

    peaks, _ = find_peaks(signal, distance=distance, prominence=prominence)
    if len(peaks) == 0:
        peaks, _ = find_peaks(np.abs(signal), distance=distance, prominence=np.std(np.abs(signal)) * 0.3)
    return peaks

def extractMedianBeat(X, leads, fs):
    leadPriority = ["II", "V2", "V1", "V3", "I"]
    pre = int(0.25 * fs)
    post = int(0.45 * fs)

    peaks = np.array([], dtype=int)
    for refLead in leadPriority:
        if refLead in leads:
            refIndex = leads.index(refLead)
            peaks = detectRPeaks(X[:, refIndex], fs)
            peaks = peaks[(peaks > pre) & (peaks < len(X) - post)]
            if len(peaks) >= 2:
                break

    beats = []
    for peak in peaks:
        beat = X[peak - pre : peak + post, :]
        if beat.shape[0] == pre + post:
            beats.append(beat)

    if len(beats) == 0:
        center = len(X) // 2
        beat = X[max(0, center - pre) : min(len(X), center + post), :]
        padded = np.zeros((pre + post, X.shape[1]), dtype=np.float32)
        padded[: beat.shape[0], :] = beat
        return padded, pre

    medianBeat = np.median(np.stack(beats, axis=0), axis=0)
    return medianBeat.astype(np.float32), pre

def buildSequenceSample(patientId, focusLeads=standardLeads):
    X, leads, fs = loadEcgRecord(patientId)
    Xf = bandpassFilter(X, fs=fs, low=0.5, high=40.0, order=3)
    Xn = robustScale(Xf)
    medianBeat, _ = extractMedianBeat(Xn, leads, fs)
    lead_indices = [leads.index(lead_name) for lead_name in focusLeads]
    return medianBeat[:, lead_indices].astype(np.float32)

def zeroCrossings(x):
    return int(np.sum(np.diff(np.signbit(x)).astype(int)))

def safeSlope(y):
    if len(y) < 2:
        return 0.0
    xs = np.arange(len(y))
    return float(np.polyfit(xs, y, 1)[0])

def integrateAbsArea(x):
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return float(trapezoid(np.abs(x)))
    return float(np.trapz(np.abs(x)))

def extractStripFeatures(x, prefix):
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_ptp": float(np.ptp(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_q05": float(np.quantile(x, 0.05)),
        f"{prefix}_q25": float(np.quantile(x, 0.25)),
        f"{prefix}_q75": float(np.quantile(x, 0.75)),
        f"{prefix}_q95": float(np.quantile(x, 0.95)),
        f"{prefix}_rms": float(np.sqrt(np.mean(x**2))),
        f"{prefix}_abs_area": float(np.sum(np.abs(x))),
        f"{prefix}_energy": float(np.sum(x**2)),
        f"{prefix}_skew": float(skew(x)),
        f"{prefix}_kurtosis": float(kurtosis(x)),
        f"{prefix}_zero_cross": float(zeroCrossings(x)),
    }

def extractBeatFeatures(beat1d, rIndex, fs, prefix):
    qWindow = beat1d[max(0, rIndex - 8) : rIndex]
    sWindow = beat1d[rIndex : min(len(beat1d), rIndex + 12)]
    stWindow = beat1d[min(len(beat1d), rIndex + 6) : min(len(beat1d), rIndex + 16)]
    tWindow = beat1d[min(len(beat1d), rIndex + 18) : min(len(beat1d), rIndex + 35)]
    preWindow = beat1d[max(0, rIndex - 10) : max(1, rIndex - 2)]

    return {
        f"{prefix}_beat_r_amp": float(beat1d[rIndex]),
        f"{prefix}_beat_q_min": float(np.min(qWindow)) if len(qWindow) else 0.0,
        f"{prefix}_beat_s_min": float(np.min(sWindow)) if len(sWindow) else 0.0,
        f"{prefix}_beat_qrs_range": float(np.ptp(beat1d[max(0, rIndex - 6) : min(len(beat1d), rIndex + 10)])),
        f"{prefix}_beat_st_mean": float(np.mean(stWindow)) if len(stWindow) else 0.0,
        f"{prefix}_beat_st_max": float(np.max(stWindow)) if len(stWindow) else 0.0,
        f"{prefix}_beat_st_slope": float(safeSlope(stWindow)) if len(stWindow) else 0.0,
        f"{prefix}_beat_t_max": float(np.max(tWindow)) if len(tWindow) else 0.0,
        f"{prefix}_beat_t_mean": float(np.mean(tWindow)) if len(tWindow) else 0.0,
        f"{prefix}_beat_pre_mean": float(np.mean(preWindow)) if len(preWindow) else 0.0,
        f"{prefix}_beat_area": integrateAbsArea(beat1d),
        f"{prefix}_beat_energy": float(np.sum(beat1d**2)),
    }

def buildFeatureRow(patientId):
    X, leads, fs = loadEcgRecord(patientId)

    Xf = bandpassFilter(X, fs=fs, low=0.5, high=40.0, order=3)
    Xn = robustScale(Xf)
    medianBeat, rIndex = extractMedianBeat(Xn, leads, fs)

    featureMap = {
        "patient_id": int(patientId),
        "fs": int(fs),
        "n_samples": int(Xn.shape[0]),
    }

    for leadIndex, leadName in enumerate(leads):
        featureMap.update(extractStripFeatures(Xn[:, leadIndex], prefix=f"{leadName}_strip"))

    for leadIndex, leadName in enumerate(leads):
        featureMap.update(extractBeatFeatures(medianBeat[:, leadIndex], rIndex=rIndex, fs=fs, prefix=f"{leadName}"))

    for leadName in ["V1", "V2", "V3"]:
        leadIndex = leads.index(leadName)
        segment = medianBeat[:, leadIndex]
        featureMap[f"{leadName}_post_r_100ms_mean"] = float(np.mean(segment[rIndex + 5 : rIndex + 15]))
        featureMap[f"{leadName}_post_r_150ms_mean"] = float(np.mean(segment[rIndex + 10 : rIndex + 20]))
        featureMap[f"{leadName}_post_r_200ms_mean"] = float(np.mean(segment[rIndex + 15 : rIndex + 25]))

    featureMap["V1_V2_st_mean_avg"] = float(np.mean([featureMap["V1_beat_st_mean"], featureMap["V2_beat_st_mean"]]))
    featureMap["V1_V3_st_mean_avg"] = float(np.mean([featureMap["V1_beat_st_mean"], featureMap["V3_beat_st_mean"]]))
    featureMap["V1_V2_qrs_avg"] = float(np.mean([featureMap["V1_beat_qrs_range"], featureMap["V2_beat_qrs_range"]]))
    featureMap["V1_V2_strip_energy_avg"] = float(np.mean([featureMap["V1_strip_energy"], featureMap["V2_strip_energy"]]))
    featureMap["V1_minus_V2_st_mean"] = float(featureMap["V1_beat_st_mean"] - featureMap["V2_beat_st_mean"])
    featureMap["V2_minus_V3_st_mean"] = float(featureMap["V2_beat_st_mean"] - featureMap["V3_beat_st_mean"])

    for lead_a, lead_b in [("V1", "V2"), ("V2", "V3"), ("I", "II"), ("II", "V1")]:
        idx_a, idx_b = leads.index(lead_a), leads.index(lead_b)
        corr = np.corrcoef(Xn[:, idx_a], Xn[:, idx_b])[0, 1]
        featureMap[f"corr_{lead_a}_{lead_b}"] = float(np.nan_to_num(corr))

    return featureMap

featureRows = []

for row in tqdm(metadata.itertuples(index=False), total=len(metadata)):
    patientId = getattr(row, "patient_id")
    try:
        featureMap = buildFeatureRow(patientId)
        featureMap["target"] = int(getattr(row, targetColumn))
        if "basal_pattern" in metadata.columns:
            featureMap["basal_pattern_meta"] = getattr(row, "basal_pattern", np.nan)
        if "sudden_death" in metadata.columns:
            featureMap["sudden_death_meta"] = getattr(row, "sudden_death", np.nan)
        featureRows.append(featureMap)
    except Exception as exc:
        print(f"Skipping patient_id={patientId} because of preprocessing error: {exc}")

featuresDf = pd.DataFrame(featureRows)
print("Feature table shape:", featuresDf.shape)
display(featuresDf.head())

missingSummary = featuresDf.isna().mean().sort_values(ascending=False)
display(missingSummary.head(20))

metadataShortcutColumns = [column for column in ["basal_pattern_meta", "sudden_death_meta"] if column in featuresDf.columns]
if metadataShortcutColumns:
    print("Metadata shortcut-risk columns detected:", metadataShortcutColumns)
    print("These columns are kept only for ablation and failure analysis. They are excluded from the primary ECG-only benchmark.")

plt.figure(figsize=(6, 4))
sns.histplot(missingSummary.values, bins=20)
plt.title("Missing Ratio Across Engineered Features")
plt.xlabel("Missing ratio")
plt.ylabel("Count")
save_plot("preprocessing_missing_ratio_histogram")
