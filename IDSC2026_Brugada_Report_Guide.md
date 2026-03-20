# IDSC 2026 Brugada-HUCA Report Guide

Dokumen ini dibuat untuk membantu penyusunan report/final write-up dari pipeline di `BinaryClassificationBrugadaHuca.ipynb`.

## 1. Framing Utama di Report

Sebelum masuk ke ranking model, tulis posisi metodologinya dengan tegas:

`This project uses ECG-only as the primary benchmark to avoid shortcut learning from metadata. Clinical metadata may be informative, but including it in the main classifier could make the model look better for the wrong reason rather than proving stronger ECG understanding.`

Kalau ingin membahas metadata, posisikan hanya sebagai analisis tambahan atau ablation, bukan sebagai model utama.

Jangan tampilkan hanya satu "winner". Report sebaiknya selalu memisahkan dua perspektif:

- `Best aggregate model`
  Gunakan model peringkat 1 dari `summary.csv`.
  Narasinya: model ini memimpin ranking agregat yang menekankan PR-AUC, F1, sensitivity, balanced accuracy, dan correct Brugada detection.

- `Best safety-oriented model`
  Gunakan model peringkat 1 dari `clinical_summary.csv`.
  Narasinya: model ini diprioritaskan untuk sensitivity yang lebih tinggi dan false negative Brugada yang lebih rendah, sehingga lebih relevan untuk pembahasan safety dan clinical relevance.

Jika kedua modelnya sama, katakan itu secara eksplisit.
Jika berbeda, katakan bahwa perbedaan ini penting dan memang diharapkan pada tugas klinis dengan biaya false negative yang tinggi.

Tambahkan satu kalimat yang blak-blakan soal risiko evaluasi:

`Because multiple models were compared on a single held-out test split, the leaderboard should be interpreted as a useful benchmark rather than a definitive estimate of out-of-sample superiority.`

Tambahkan juga satu kalimat yang jelas soal fairness feature set:

`The main comparison excludes metadata such as basal_pattern and sudden_death so that the reported benchmark remains ECG-driven, fair, and clinically defensible.`

## 2. Cara Menulis Pemilihan Threshold

Jangan pitch model final hanya dari PR-AUC atau F1.
Gunakan bahasa seperti ini:

`Threshold dipilih pada validation split, bukan test split, untuk menghindari optimistic bias. Pemilihan threshold diarahkan untuk menjaga sensitivity Brugada tetap tinggi sambil membatasi false positives ke tingkat yang masih dapat diterima untuk workflow screening.`

Di report utama, tampilkan tabel atau paragraf yang mencakup:

- threshold terpilih
- sensitivity pada threshold itu
- specificity pada threshold itu
- jumlah false positives
- jumlah false negatives
- alasan klinis pemilihan threshold

Gunakan artefak berikut dari notebook:

- `predict/clinical_candidate_threshold_scan.csv`
- `plots/failure_analysis_clinical_candidate_threshold_tradeoff.png`

## 3. Interpretability yang Harus Masuk ke Report

PhysioNet menjelaskan bahwa Brugada syndrome berhubungan dengan abnormalitas ECG khas, termasuk coved-type ST-segment elevation pada lead `V1–V3`.

Karena itu, report sebaiknya menghubungkan model ke area tersebut lewat dua jalur:

- `Feature story`
  Gunakan permutation importance untuk menunjukkan fitur ECG mana yang paling berkontribusi.
  Prioritaskan narasi seputar fitur dari lead `V1`, `V2`, dan `V3`.

- `Visual story`
  Gunakan saliency/heatmap pada model sequence untuk menunjukkan apakah perhatian model terkonsentrasi di lead `V1–V3`.

Artefak yang sebaiknya dimasukkan ke report:

- `predict/feature_permutation_importance.csv`
- `predict/feature_permutation_importance_by_lead.csv`
- `plots/interpretability_feature_permutation_importance.png`
- `plots/interpretability_feature_importance_by_lead.png`
- `predict/sequence_lead_saliency.csv`
- `plots/interpretability_sequence_lead_saliency.png`
- `plots/interpretability_sequence_v1_v3_saliency_heatmap.png`

Kalimat report yang aman:

`Interpretability analysis menunjukkan bahwa model memanfaatkan informasi yang konsisten dengan lead prekordial kanan, terutama V1–V3, yang memang dilaporkan sebagai area ECG paling relevan untuk Brugada syndrome.`

Jangan klaim bahwa saliency membuktikan kausalitas. Presentasikan sebagai transparansi model, bukan bukti mekanistik final.

Kalimat report yang lebih aman:

`These interpretability outputs should be read as descriptive transparency tools. They help assess whether the model is focusing on clinically plausible ECG regions, but they do not prove causal reasoning or guarantee safe deployment.`

## 4. Validation Rigor yang Harus Masuk ke Report Utama

Repeated split results jangan dibiarkan hanya ada di notebook.
Masukkan ke report utama dalam bentuk tabel ringkas `mean ± std`.

Gunakan file berikut:

- `predict/stability_repeated_split_summary.csv`
- `predict/stability_repeated_split_details.csv`
- `plots/validation_repeated_split_sensitivity_boxplot.png`

Poin yang perlu ditulis:

- evaluasi dilakukan dengan repeated patient-level stratified splits pada pool train+validation
- hasil digunakan untuk menilai stabilitas sensitivity, PR-AUC, dan F1
- analisis ini tidak menggantikan external validation, tetapi memperkuat rigor dibanding satu split tunggal

Kalimat report yang aman:

`To assess robustness on a small and imbalanced cohort, we supplemented the shared hold-out benchmark with repeated patient-level stratified splits on the train/validation pool and report mean ± standard deviation for the main candidate models.`

Tambahkan juga satu kalimat pendamping:

`This added stability analysis improves rigor, but it does not fully eliminate the model-selection optimism that can arise when many candidate models are compared against one final hold-out split.`

## 5. Tabel dan Figure yang Paling Bernilai

Minimal masukkan ini ke report:

- satu tabel `Best aggregate model` vs `Best safety-oriented model`
- satu tabel repeated split `mean ± std`
- satu figure threshold trade-off
- satu figure confusion matrix untuk safety-oriented model
- satu figure interpretability feature-based
- satu figure saliency V1–V3
- satu paragraf failure analysis yang menyebut false negative Brugada

## 6. Failure Analysis yang Perlu Ditulis

Report harus menyebut:

- siapa model safety-oriented yang dipilih
- berapa false negative Brugada yang masih tersisa
- mengapa ini tetap jadi limitasi penting
- bahwa model belum layak sebagai autonomous diagnostic tool
- apakah model safety-oriented tersebut menangkap lebih banyak kasus Brugada dibanding leaderboard winner
- berapa harga yang dibayar dalam bentuk false positives tambahan

Artefak yang bisa dipakai:

- `predict/clinical_candidate_failure_overview.csv`
- `predict/clinical_candidate_false_negatives.csv`
- `plots/failure_false_negative_patient_*_v1_v3.png`
- `plots/failure_analysis_basal_pattern_vs_outcome.png`

Kalau artefak metadata ikut ditampilkan, beri framing yang jelas:

`Metadata-based patterns are presented only as a supplementary sensitivity analysis. They are not part of the primary ECG-only benchmark and should not be used to claim that the main classifier is better.`

## 7. Sitasi Dataset yang Harus Ada

Masukkan keduanya:

### Dataset-specific citation

Costa Cortez, N., & Garcia Iglesias, D. (2026). *Brugada-HUCA: 12-Lead ECG Recordings for the Study of Brugada Syndrome* (version 1.0.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/0m2w-dy83

### Standard PhysioNet citation

Goldberger A, Amaral L, Glass L, Hausdorff J, Ivanov PC, Mark R, Mietus JE, Moody GB, Peng CK, Stanley HE. *PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals*. Circulation [Online]. 2000;101(23):e215-e220. RRID:SCR_007345. doi:10.1161/01.CIR.101.23.e215

Kalau format report memakai APA/Vancouver/IEEE, ubah gaya penulisan, tetapi jangan hilangkan:

- nama dataset
- versi dataset
- PhysioNet
- DOI dataset
- standard PhysioNet citation

Sumber resmi sitasi:

- Dataset page: `https://physionet.org/content/brugada-huca/1.0.0/`
- PhysioNet citation guidance: `https://archive.physionet.org/citations.shtml`

## 8. Checklist Sebelum Submit

- `summary.csv` sudah dipakai untuk `best aggregate model`
- `clinical_summary.csv` sudah dipakai untuk `best safety-oriented model`
- threshold rationale ditulis jelas
- sensitivity vs false positives dibahas eksplisit
- interpretability V1–V3 masuk ke report
- repeated split `mean ± std` masuk ke report utama
- failure analysis false negative masuk ke report
- limitasi single-center, small sample, imbalance, dan no external validation ditulis
- sitasi dataset spesifik dan sitasi standar PhysioNet keduanya tercantum
