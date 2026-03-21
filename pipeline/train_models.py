def specificityScore(yTrue, yPred):
    yTrue = np.asarray(yTrue).astype(int)
    yPred = np.asarray(yPred).astype(int)
    matrix = confusion_matrix(yTrue, yPred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def sensitivityScore(yTrue, yPred):
    yTrue = np.asarray(yTrue).astype(int)
    yPred = np.asarray(yPred).astype(int)
    matrix = confusion_matrix(yTrue, yPred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def tuneThreshold(yTrue, yProb):
    yTrue = np.asarray(yTrue).astype(int)
    yProb = np.asarray(yProb).astype(float)

    precisionValues, recallValues, thresholdValues = precision_recall_curve(yTrue, yProb)
    if len(thresholdValues) == 0:
        defaultThreshold = 0.5
        defaultF1 = f1_score(yTrue, (yProb >= defaultThreshold).astype(int))
        return float(defaultThreshold), float(defaultF1)

    precisionValues = precisionValues[:-1]
    recallValues = recallValues[:-1]
    f1Values = 2 * precisionValues * recallValues / np.clip(precisionValues + recallValues, 1e-8, None)
    bestIndex = int(np.nanargmax(f1Values))
    return float(thresholdValues[bestIndex]), float(f1Values[bestIndex])

def classCountReport(yTrue, yPred, modelName):
    yTrue = np.asarray(yTrue).astype(int)
    yPred = np.asarray(yPred).astype(int)
    rows = []
    for cls, className in [(0, "Normal"), (1, "Brugada")]:
        actualCount = int(np.sum(yTrue == cls))
        predictedCount = int(np.sum(yPred == cls))
        correctCount = int(np.sum((yTrue == cls) & (yPred == cls)))
        rows.append(
            {
                "Model": modelName,
                "Class": f"{className} ({cls})",
                "Actual Count": actualCount,
                "Predicted Count": predictedCount,
                "Correctly Predicted": correctCount,
                "Recall per Class": correctCount / actualCount if actualCount > 0 else np.nan,
                "Precision on Predicted Class": correctCount / predictedCount if predictedCount > 0 else np.nan,
            }
        )
    rows.append(
        {
            "Model": modelName,
            "Class": "TOTAL",
            "Actual Count": int(len(yTrue)),
            "Predicted Count": int(len(yPred)),
            "Correctly Predicted": int(np.sum(yTrue == yPred)),
            "Recall per Class": float(np.mean(yTrue == yPred)),
            "Precision on Predicted Class": np.nan,
        }
    )
    return pd.DataFrame(rows)

def calculateBinaryMetrics(yTrue, yProb, threshold):
    yTrue = np.asarray(yTrue).astype(int)
    yProb = np.asarray(yProb).astype(float)
    yPred = (yProb >= threshold).astype(int)

    matrix = confusion_matrix(yTrue, yPred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    metrics = {
        "Threshold": float(threshold),
        "AUC": float(roc_auc_score(yTrue, yProb)) if len(np.unique(yTrue)) > 1 else np.nan,
        "PR AUC": float(average_precision_score(yTrue, yProb)) if len(np.unique(yTrue)) > 1 else np.nan,
        "F1": float(f1_score(yTrue, yPred)),
        "Accuracy": float(accuracy_score(yTrue, yPred)),
        "Balanced Accuracy": float(balanced_accuracy_score(yTrue, yPred)),
        "Sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "Specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "Precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "Correctly Predicted": int(np.sum(yTrue == yPred)),
        "Total Samples": int(len(yTrue)),
        "Correct Normal": int(tn),
        "Correct Brugada": int(tp),
        "False Positives": int(fp),
        "False Negatives": int(fn),
        "True Positives": int(tp),
        "True Negatives": int(tn),
    }
    return metrics, yPred

def evaluateThresholdGrid(yTrue, yProb, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    for threshold in thresholds:
        metricRow, _ = calculateBinaryMetrics(yTrue, yProb, threshold)
        rows.append(metricRow)
    return pd.DataFrame(rows)

def evaluateBinaryPredictions(modelName, family, patientIds, yTrue, yProb, threshold):
    baseMetrics, yPred = calculateBinaryMetrics(yTrue, yProb, threshold)

    predictionDf = pd.DataFrame(
        {
            "patient_id": [int(pid) for pid in patientIds],
            "target": np.asarray(yTrue).astype(int),
            "pred_proba": np.asarray(yProb).astype(float),
            "pred_label": yPred,
        }
    )
    predictionFile = predictionDir / make_filename(f"{modelName} Predictions", "csv")
    predictionDf.to_csv(predictionFile, index=False)

    metrics = {
        "Model": modelName,
        "Family": family,
        **baseMetrics,
        "Prediction CSV": repoDisplayPath(predictionFile),
    }
    countDf = classCountReport(yTrue, yPred, modelName)
    return metrics, predictionDf, countDf

def makeClassWeightDict(y):
    y = np.asarray(y).astype(int)
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}

def makeSampleWeightVector(y):
    classWeight = makeClassWeightDict(y)
    y = np.asarray(y).astype(int)
    return np.array([classWeight[int(label)] for label in y], dtype=float)

maxTrainingEpochs = 200
maxPretrainEpochs = 200
earlyStoppingPatience = 20
pretrainStoppingPatience = 20
vicregPretrainPatience = 15

def makeTrainingCallbacks(monitor="val_auc", mode="max", patience=earlyStoppingPatience, minDelta=1e-4):
    return [
        callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            min_delta=minDelta,
            restore_best_weights=True,
            verbose=1,
        )
    ]

def historyToDict(history):
    return {key: [float(v) for v in values] for key, values in history.history.items()}

def recordTrainingFailure(modelName, family, exc):
    failureRow = {
        "Model": modelName,
        "Family": family,
        "Error Type": type(exc).__name__,
        "Error Message": str(exc),
    }
    benchmarkFailures.append(failureRow)
    print(f"{modelName} failed with {type(exc).__name__}: {exc}")
    print(traceback.format_exc())
    return failureRow

leaderboardSortColumns = [
    "PR AUC",
    "F1",
    "Sensitivity",
    "Correct Brugada",
    "Balanced Accuracy",
    "AUC",
    "Specificity",
    "Accuracy",
    "Correctly Predicted",
]
clinicalSortColumns = [
    "Sensitivity",
    "Correct Brugada",
    "False Negatives",
    "PR AUC",
    "F1",
    "Balanced Accuracy",
    "Specificity",
    "AUC",
    "Accuracy",
]
clinicalAscending = [False, False, True, False, False, False, False, False, False]

modelArtifactDir = predictionDir / "models"
modelArtifactDir.mkdir(exist_ok=True)

benchmarkResults = []
benchmarkCountReports = []
benchmarkPredictionTables = {}
benchmarkHistories = {}
benchmarkFailures = []
trainedClassicalModels = {}
trainedSequenceModelArtifacts = {}

scalePosWeightBenchmark = (len(yTrainTab) - yTrainTab.sum()) / max(yTrainTab.sum(), 1)
trainClassWeightMap = makeClassWeightDict(yTrainTab)
trainSampleWeightVector = makeSampleWeightVector(yTrainTab)

print("Training class weights:", trainClassWeightMap)
print("scale_pos_weight for XGBoost:", float(scalePosWeightBenchmark))

classicalModels = {
    "XGBoost Features": XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.3,
        reg_lambda=1.5,
        min_child_weight=2,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=randomState,
        scale_pos_weight=scalePosWeightBenchmark,
        tree_method="hist",
    ),
    "Random Forest Features": RandomForestClassifier(
        n_estimators=600,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=randomState,
        n_jobs=-1,
    ),
    "SVM RBF Features": SVC(
        kernel="rbf",
        C=2.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=randomState,
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=350,
        learning_rate=0.05,
        random_state=randomState,
    ),
}

for modelName, model in classicalModels.items():
    print()
    print(f"Training {modelName} on ECG-only features ...")

    try:
        if "SVM" in modelName:
            xTrainInput, xValInput, xTestInput = xTrainTabScaled, xValTabScaled, xTestTabScaled
        else:
            xTrainInput, xValInput, xTestInput = xTrainTab, xValTab, xTestTab

        if modelName == "AdaBoost":
            model.fit(xTrainInput, yTrainTab, sample_weight=trainSampleWeightVector)
        else:
            model.fit(xTrainInput, yTrainTab)

        valProba = model.predict_proba(xValInput)[:, 1].astype(float)
        bestThreshold, bestValF1 = tuneThreshold(yValTab, valProba)
        testProba = model.predict_proba(xTestInput)[:, 1].astype(float)

        modelMetrics, predictionDf, countDf = evaluateBinaryPredictions(
            modelName=modelName,
            family="Feature-Based",
            patientIds=testIds,
            yTrue=yTestTab,
            yProb=testProba,
            threshold=bestThreshold,
        )
        modelMetrics["Validation F1"] = float(bestValF1)
        modelMetrics["Feature Set"] = "ECG-only primary benchmark"

        benchmarkResults.append(modelMetrics)
        benchmarkCountReports.append(countDf)
        benchmarkPredictionTables[modelName] = predictionDf
        trainedClassicalModels[modelName] = model

        print(
            f"{modelName} | PR-AUC={modelMetrics['PR AUC']:.4f} | F1={modelMetrics['F1']:.4f} | "
            f"Sensitivity={modelMetrics['Sensitivity']:.4f} | False Negatives={modelMetrics['False Negatives']}"
        )
        print("Prediction CSV:", modelMetrics["Prediction CSV"])
    except Exception as exc:
        recordTrainingFailure(modelName, "Feature-Based", exc)

def residualBlock(x, filters, kernelSize=3, stride=1, dropoutRate=0.10):
    shortcut = x
    if stride != 1 or int(x.shape[-1]) != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.Conv1D(filters, kernelSize, strides=stride, padding="same")(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv1D(filters, kernelSize, padding="same")(y)
    y = layers.BatchNormalization()(y)
    if dropoutRate > 0:
        y = layers.SpatialDropout1D(dropoutRate)(y)

    out = layers.Add()([shortcut, y])
    out = layers.Activation("relu")(out)
    return out

def buildCnn1d(inputShape, learningRate=1e-3):
    inp = layers.Input(shape=inputShape)
    x = layers.Conv1D(32, 7, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out, name="cnn_1d")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learningRate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
        jit_compile=False,
    )
    return model

def buildResnet1d(inputShape, learningRate=1e-3):
    inp = layers.Input(shape=inputShape)
    x = layers.Conv1D(32, 7, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = residualBlock(x, 32, stride=1, dropoutRate=0.05)
    x = residualBlock(x, 64, stride=2, dropoutRate=0.10)
    x = residualBlock(x, 128, stride=2, dropoutRate=0.10)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out, name="resnet_1d")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learningRate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
        jit_compile=False,
    )
    return model

def buildCnnBigru(inputShape, learningRate=1e-3):
    inp = layers.Input(shape=inputShape)
    x = layers.Conv1D(32, 5, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.GRU(32))(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out, name="cnn_bigru")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learningRate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
        jit_compile=False,
    )
    return model

class LearnablePositionEmbedding(layers.Layer):
    def __init__(self, sequenceLength, modelDim, **kwargs):
        super().__init__(**kwargs)
        self.sequenceLength = int(sequenceLength)
        self.modelDim = int(modelDim)
        self.positionEmbedding = layers.Embedding(input_dim=self.sequenceLength, output_dim=self.modelDim)

    def call(self, inputs):
        positionIndex = tf.range(start=0, limit=self.sequenceLength, delta=1)
        positionEncoding = self.positionEmbedding(positionIndex)
        positionEncoding = tf.expand_dims(positionEncoding, axis=0)
        return inputs + positionEncoding

def transformerBlock(x, numHeads=4, ffDim=128, dropoutRate=0.10):
    attn = layers.MultiHeadAttention(
        num_heads=numHeads,
        key_dim=max(8, int(x.shape[-1]) // numHeads),
        dropout=dropoutRate,
    )(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    ff = layers.Dense(ffDim, activation="relu")(x)
    ff = layers.Dropout(dropoutRate)(ff)
    ff = layers.Dense(int(x.shape[-1]))(ff)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x

def buildTransformerModel(inputShape, learningRate=8e-4):
    inp = layers.Input(shape=inputShape)
    modelDim = 64
    x = layers.Dense(modelDim)(inp)
    x = LearnablePositionEmbedding(sequenceLength=inputShape[0], modelDim=modelDim)(x)
    x = transformerBlock(x, numHeads=4, ffDim=128, dropoutRate=0.10)
    x = transformerBlock(x, numHeads=4, ffDim=128, dropoutRate=0.10)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out, name="transformer_1d")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learningRate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
        jit_compile=False,
    )
    return model

def trainDeepClassifier(modelName, modelBuilder, xTrain, yTrain, xVal, yVal, xTest, yTest, testPatientIds, learningRate=1e-3, epochs=maxTrainingEpochs, batchSize=16):
    tf.keras.backend.clear_session()
    gc.collect()
    set_global_seed(randomState)

    xTrain = np.asarray(xTrain).astype(np.float32)
    xVal = np.asarray(xVal).astype(np.float32)
    xTest = np.asarray(xTest).astype(np.float32)
    yTrain = np.asarray(yTrain).astype(np.float32).ravel()
    yVal = np.asarray(yVal).astype(np.float32).ravel()
    yTest = np.asarray(yTest).astype(np.float32).ravel()

    model = modelBuilder(inputShape=xTrain.shape[1:], learningRate=learningRate)
    classWeight = makeClassWeightDict(yTrain)

    history = model.fit(
        xTrain,
        yTrain,
        validation_data=(xVal, yVal),
        epochs=epochs,
        batch_size=batchSize,
        callbacks=makeTrainingCallbacks(monitor="val_auc", mode="max", patience=earlyStoppingPatience),
        class_weight=classWeight,
        verbose=2,
    )

    valProba = model.predict(xVal, verbose=0).ravel()
    bestThreshold, bestValF1 = tuneThreshold(yVal, valProba)
    testProba = model.predict(xTest, verbose=0).ravel()

    modelMetrics, predictionDf, countDf = evaluateBinaryPredictions(
        modelName=modelName,
        family="Deep Learning End-to-End",
        patientIds=testPatientIds,
        yTrue=yTest,
        yProb=testProba,
        threshold=bestThreshold,
    )
    modelMetrics["Validation F1"] = float(bestValF1)
    modelMetrics["Epochs Trained"] = int(len(history.history.get("loss", [])))
    modelMetrics["Feature Set"] = "ECG-only primary benchmark"

    modelArtifactPath = modelArtifactDir / make_filename(modelName, "keras")
    model.save(modelArtifactPath)
    trainedSequenceModelArtifacts[modelName] = str(modelArtifactPath.resolve())
    modelMetrics["Model Artifact"] = repoDisplayPath(modelArtifactPath)

    historyDict = historyToDict(history)

    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return modelMetrics, predictionDf, countDf, historyDict

deepModelBuilders = [
    ("1D CNN Median Beat", buildCnn1d, 1e-3),
    ("ResNet 1D Median Beat", buildResnet1d, 1e-3),
    ("CNN + BiGRU Median Beat", buildCnnBigru, 1e-3),
    ("Transformer Encoder Median Beat", buildTransformerModel, 8e-4),
]

for modelName, modelBuilder, learningRate in deepModelBuilders:
    print(f"\nTraining {modelName} ...")
    try:
        modelMetrics, predictionDf, countDf, historyDict = trainDeepClassifier(
            modelName=modelName,
            modelBuilder=modelBuilder,
            xTrain=xTrainSeq,
            yTrain=yTrainSeq,
            xVal=xValSeq,
            yVal=yValSeq,
            xTest=xTestSeq,
            yTest=yTestSeq,
            testPatientIds=testIds,
            learningRate=learningRate,
            epochs=maxTrainingEpochs,
            batchSize=16,
        )
        benchmarkResults.append(modelMetrics)
        benchmarkCountReports.append(countDf)
        benchmarkPredictionTables[modelName] = predictionDf
        benchmarkHistories[modelName] = historyDict

        print(
            f"{modelName} | AUC={modelMetrics['AUC']:.4f} | F1={modelMetrics['F1']:.4f} | "
            f"Sensitivity={modelMetrics['Sensitivity']:.4f} | False Negatives={modelMetrics['False Negatives']}"
        )
        print("Prediction CSV:", modelMetrics["Prediction CSV"])
    except Exception as exc:
        recordTrainingFailure(modelName, "Deep Learning End-to-End", exc)

def buildTransferEncoder(inputShape):
    inp = layers.Input(shape=inputShape)
    x = layers.Conv1D(32, 5, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2, padding="same")(x)
    x = residualBlock(x, 64, stride=1, dropoutRate=0.05)
    x = layers.MaxPooling1D(2, padding="same")(x)
    x = residualBlock(x, 128, stride=1, dropoutRate=0.10)
    return models.Model(inp, x, name="transfer_encoder")

def buildDenoisingAutoencoder(inputShape):
    encoder = buildTransferEncoder(inputShape)
    inp = layers.Input(shape=inputShape)
    x = layers.GaussianNoise(0.05)(inp)
    x = encoder(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32, 3, padding="same", activation="relu")(x)
    x = layers.Conv1D(inputShape[-1], 1, padding="same")(x)

    currentLen = int(x.shape[1])
    targetLen = inputShape[0]
    if currentLen > targetLen:
        cropTotal = currentLen - targetLen
        cropLeft = cropTotal // 2
        cropRight = cropTotal - cropLeft
        x = layers.Cropping1D((cropLeft, cropRight))(x)
    elif currentLen < targetLen:
        padTotal = targetLen - currentLen
        padLeft = padTotal // 2
        padRight = padTotal - padLeft
        x = layers.ZeroPadding1D((padLeft, padRight))(x)

    autoencoder = models.Model(inp, x, name="denoising_autoencoder")
    autoencoder.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss="mse", jit_compile=False)
    return encoder, autoencoder

def buildTransferClassifier(encoder, inputShape, learningRate=3e-4):
    inp = layers.Input(shape=inputShape)
    x = encoder(inp)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out, name="transfer_classifier")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learningRate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
        jit_compile=False,
    )
    return model

def trainTransferLearningModel(modelName, xTrain, yTrain, xVal, yVal, xTest, yTest, testPatientIds, pretrainEpochs=maxPretrainEpochs, finetuneEpochs=maxTrainingEpochs, batchSize=16):
    tf.keras.backend.clear_session()
    gc.collect()
    set_global_seed(randomState)

    xTrain = np.asarray(xTrain).astype(np.float32)
    xVal = np.asarray(xVal).astype(np.float32)
    xTest = np.asarray(xTest).astype(np.float32)
    yTrain = np.asarray(yTrain).astype(np.float32).ravel()
    yVal = np.asarray(yVal).astype(np.float32).ravel()
    yTest = np.asarray(yTest).astype(np.float32).ravel()

    encoder, autoencoder = buildDenoisingAutoencoder(inputShape=xTrain.shape[1:])
    pretrainHistory = autoencoder.fit(
        xTrain,
        xTrain,
        validation_data=(xVal, xVal),
        epochs=pretrainEpochs,
        batch_size=batchSize,
        callbacks=makeTrainingCallbacks(monitor="val_loss", mode="min", patience=pretrainStoppingPatience),
        verbose=2,
    )

    classifier = buildTransferClassifier(
        encoder=encoder,
        inputShape=xTrain.shape[1:],
        learningRate=3e-4,
    )
    classWeight = makeClassWeightDict(yTrain)
    finetuneHistory = classifier.fit(
        xTrain,
        yTrain,
        validation_data=(xVal, yVal),
        epochs=finetuneEpochs,
        batch_size=batchSize,
        callbacks=makeTrainingCallbacks(monitor="val_auc", mode="max", patience=earlyStoppingPatience),
        class_weight=classWeight,
        verbose=2,
    )

    valProba = classifier.predict(xVal, verbose=0).ravel()
    bestThreshold, bestValF1 = tuneThreshold(yVal, valProba)
    testProba = classifier.predict(xTest, verbose=0).ravel()

    modelMetrics, predictionDf, countDf = evaluateBinaryPredictions(
        modelName=modelName,
        family="Transfer Learning / Advanced",
        patientIds=testPatientIds,
        yTrue=yTest,
        yProb=testProba,
        threshold=bestThreshold,
    )
    modelMetrics["Validation F1"] = float(bestValF1)
    modelMetrics["Pretrain Epochs"] = int(len(pretrainHistory.history.get("loss", [])))
    modelMetrics["Epochs Trained"] = int(len(finetuneHistory.history.get("loss", [])))
    modelMetrics["Feature Set"] = "ECG-only primary benchmark"

    modelArtifactPath = modelArtifactDir / make_filename(modelName, "keras")
    classifier.save(modelArtifactPath)
    trainedSequenceModelArtifacts[modelName] = str(modelArtifactPath.resolve())
    modelMetrics["Model Artifact"] = repoDisplayPath(modelArtifactPath)

    historyDict = {
        "pretrain": historyToDict(pretrainHistory),
        "finetune": historyToDict(finetuneHistory),
    }

    del autoencoder
    del classifier
    tf.keras.backend.clear_session()
    gc.collect()
    return modelMetrics, predictionDf, countDf, historyDict

def buildVicRegEncoder(inputShape, embeddingDim=64):
    inp = layers.Input(shape=inputShape)
    x = layers.Conv1D(32, 5, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = residualBlock(x, 64, stride=2, dropoutRate=0.05)
    x = residualBlock(x, 128, stride=2, dropoutRate=0.10)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(embeddingDim)(x)
    return models.Model(inp, out, name="vicreg_encoder")

def augmentEcgBatch(batch):
    batch = tf.cast(batch, tf.float32)
    noise = tf.random.normal(tf.shape(batch), stddev=0.03)
    scale = tf.random.uniform((tf.shape(batch)[0], 1, 1), minval=0.90, maxval=1.10)
    mask = tf.cast(
        tf.random.uniform((tf.shape(batch)[0], tf.shape(batch)[1], 1), minval=0.0, maxval=1.0) > 0.08,
        tf.float32,
    )
    return (batch * scale + noise) * mask

def vicregLoss(z1, z2, simCoeff=25.0, stdCoeff=25.0, covCoeff=1.0):
    z1 = tf.cast(z1, tf.float32)
    z2 = tf.cast(z2, tf.float32)

    reprLoss = tf.reduce_mean(tf.square(z1 - z2))

    z1Centered = z1 - tf.reduce_mean(z1, axis=0)
    z2Centered = z2 - tf.reduce_mean(z2, axis=0)

    stdZ1 = tf.sqrt(tf.math.reduce_variance(z1Centered, axis=0) + 1e-4)
    stdZ2 = tf.sqrt(tf.math.reduce_variance(z2Centered, axis=0) + 1e-4)
    stdLoss = tf.reduce_mean(tf.nn.relu(1.0 - stdZ1)) + tf.reduce_mean(tf.nn.relu(1.0 - stdZ2))

    batchSize = tf.cast(tf.shape(z1Centered)[0], tf.float32)
    covZ1 = tf.matmul(z1Centered, z1Centered, transpose_a=True) / tf.maximum(batchSize - 1.0, 1.0)
    covZ2 = tf.matmul(z2Centered, z2Centered, transpose_a=True) / tf.maximum(batchSize - 1.0, 1.0)

    dim = tf.shape(covZ1)[0]
    offDiagMask = tf.ones((dim, dim), dtype=tf.float32) - tf.eye(dim, dtype=tf.float32)
    covLoss = tf.reduce_mean(tf.square(covZ1 * offDiagMask)) + tf.reduce_mean(tf.square(covZ2 * offDiagMask))

    return simCoeff * reprLoss + stdCoeff * stdLoss + covCoeff * covLoss

def pretrainVicRegEncoder(xTrain, xVal, epochs=maxPretrainEpochs, batchSize=16, learningRate=1e-3, patience=vicregPretrainPatience):
    set_global_seed(randomState)
    encoder = buildVicRegEncoder(xTrain.shape[1:])
    optimizer = optimizers.Adam(learning_rate=learningRate)

    trainDs = (
        tf.data.Dataset.from_tensor_slices(np.asarray(xTrain).astype(np.float32))
        .shuffle(len(xTrain), seed=randomState, reshuffle_each_iteration=True)
        .batch(batchSize)
    )
    valDs = tf.data.Dataset.from_tensor_slices(np.asarray(xVal).astype(np.float32)).batch(batchSize)

    history = {"loss": [], "val_loss": []}
    bestWeights = encoder.get_weights()
    bestVal = np.inf
    wait = 0

    for epoch in range(epochs):
        trainLosses = []
        for batch in trainDs:
            with tf.GradientTape() as tape:
                view1 = augmentEcgBatch(batch)
                view2 = augmentEcgBatch(batch)
                z1 = encoder(view1, training=True)
                z2 = encoder(view2, training=True)
                loss = vicregLoss(z1, z2)
            gradients = tape.gradient(loss, encoder.trainable_weights)
            optimizer.apply_gradients(zip(gradients, encoder.trainable_weights))
            trainLosses.append(float(loss.numpy()))

        valLosses = []
        for batch in valDs:
            view1 = augmentEcgBatch(batch)
            view2 = augmentEcgBatch(batch)
            z1 = encoder(view1, training=False)
            z2 = encoder(view2, training=False)
            valLosses.append(float(vicregLoss(z1, z2).numpy()))

        epochTrainLoss = float(np.mean(trainLosses))
        epochValLoss = float(np.mean(valLosses))
        history["loss"].append(epochTrainLoss)
        history["val_loss"].append(epochValLoss)
        print(f"VICReg pretrain epoch {epoch + 1:02d} | loss={epochTrainLoss:.4f} | val_loss={epochValLoss:.4f}")

        if epochValLoss < bestVal - 1e-4:
            bestVal = epochValLoss
            bestWeights = encoder.get_weights()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("VICReg pretraining early stop.")
                break

    encoder.set_weights(bestWeights)
    return encoder, history

def buildVicRegClassifier(encoder, inputShape, learningRate=3e-4):
    inp = layers.Input(shape=inputShape)
    x = encoder(inp)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out, name="vicreg_classifier")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learningRate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
        jit_compile=False,
    )
    return model

def trainVicRegModel(modelName, xTrain, yTrain, xVal, yVal, xTest, yTest, testPatientIds, pretrainEpochs=maxPretrainEpochs, finetuneEpochs=maxTrainingEpochs, batchSize=16):
    tf.keras.backend.clear_session()
    gc.collect()
    set_global_seed(randomState)

    xTrain = np.asarray(xTrain).astype(np.float32)
    xVal = np.asarray(xVal).astype(np.float32)
    xTest = np.asarray(xTest).astype(np.float32)
    yTrain = np.asarray(yTrain).astype(np.float32).ravel()
    yVal = np.asarray(yVal).astype(np.float32).ravel()
    yTest = np.asarray(yTest).astype(np.float32).ravel()

    encoder, pretrainHistory = pretrainVicRegEncoder(
        xTrain=xTrain,
        xVal=xVal,
        epochs=pretrainEpochs,
        batchSize=batchSize,
        learningRate=1e-3,
        patience=vicregPretrainPatience,
    )
    classifier = buildVicRegClassifier(encoder=encoder, inputShape=xTrain.shape[1:], learningRate=3e-4)
    classWeight = makeClassWeightDict(yTrain)
    finetuneHistory = classifier.fit(
        xTrain,
        yTrain,
        validation_data=(xVal, yVal),
        epochs=finetuneEpochs,
        batch_size=batchSize,
        callbacks=makeTrainingCallbacks(monitor="val_auc", mode="max", patience=earlyStoppingPatience),
        class_weight=classWeight,
        verbose=2,
    )

    valProba = classifier.predict(xVal, verbose=0).ravel()
    bestThreshold, bestValF1 = tuneThreshold(yVal, valProba)
    testProba = classifier.predict(xTest, verbose=0).ravel()

    modelMetrics, predictionDf, countDf = evaluateBinaryPredictions(
        modelName=modelName,
        family="Transfer Learning / Advanced",
        patientIds=testPatientIds,
        yTrue=yTest,
        yProb=testProba,
        threshold=bestThreshold,
    )
    modelMetrics["Validation F1"] = float(bestValF1)
    modelMetrics["Pretrain Epochs"] = int(len(pretrainHistory["loss"]))
    modelMetrics["Epochs Trained"] = int(len(finetuneHistory.history.get("loss", [])))
    modelMetrics["Feature Set"] = "ECG-only primary benchmark"

    modelArtifactPath = modelArtifactDir / make_filename(modelName, "keras")
    classifier.save(modelArtifactPath)
    trainedSequenceModelArtifacts[modelName] = str(modelArtifactPath.resolve())
    modelMetrics["Model Artifact"] = repoDisplayPath(modelArtifactPath)

    historyDict = {
        "pretrain": pretrainHistory,
        "finetune": historyToDict(finetuneHistory),
    }

    del classifier
    tf.keras.backend.clear_session()
    gc.collect()
    return modelMetrics, predictionDf, countDf, historyDict

def esnTransform(X, nReservoir=160, spectralRadius=0.90, leakRate=0.35, seed=42):
    rng = np.random.default_rng(seed)
    X = np.asarray(X).astype(np.float32)
    nInputs = X.shape[-1]
    win = rng.uniform(-0.5, 0.5, size=(nInputs, nReservoir)).astype(np.float32)
    w = rng.uniform(-0.5, 0.5, size=(nReservoir, nReservoir)).astype(np.float32)
    eigenvalues = np.linalg.eigvals(w)
    spectralNorm = np.max(np.abs(eigenvalues))
    w *= spectralRadius / max(float(spectralNorm), 1e-6)
    bias = rng.uniform(-0.1, 0.1, size=(nReservoir,)).astype(np.float32)

    featureRows = []
    for sample in X:
        state = np.zeros(nReservoir, dtype=np.float32)
        stateHistory = []
        for timeIndex in range(sample.shape[0]):
            inputVector = sample[timeIndex]
            preActivation = inputVector @ win + state @ w + bias
            state = (1.0 - leakRate) * state + leakRate * np.tanh(preActivation)
            stateHistory.append(state.copy())
        stateHistory = np.stack(stateHistory, axis=0)
        featureRows.append(
            np.concatenate(
                [
                    stateHistory.mean(axis=0),
                    stateHistory.std(axis=0),
                    stateHistory[-1],
                ],
                axis=0,
            )
        )
    return np.stack(featureRows).astype(np.float32)

def trainEsnModel(modelName, xTrain, yTrain, xVal, yVal, xTest, yTest, testPatientIds):
    xTrainReservoir = esnTransform(xTrain, seed=randomState)
    xValReservoir = esnTransform(xVal, seed=randomState)
    xTestReservoir = esnTransform(xTest, seed=randomState)

    scaler = StandardScaler()
    xTrainReservoir = scaler.fit_transform(xTrainReservoir)
    xValReservoir = scaler.transform(xValReservoir)
    xTestReservoir = scaler.transform(xTestReservoir)

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=randomState,
    )
    classifier.fit(xTrainReservoir, yTrain)
    valProba = classifier.predict_proba(xValReservoir)[:, 1].astype(float)
    bestThreshold, bestValF1 = tuneThreshold(yVal, valProba)
    testProba = classifier.predict_proba(xTestReservoir)[:, 1].astype(float)

    modelMetrics, predictionDf, countDf = evaluateBinaryPredictions(
        modelName=modelName,
        family="Transfer Learning / Advanced",
        patientIds=testPatientIds,
        yTrue=yTest,
        yProb=testProba,
        threshold=bestThreshold,
    )
    modelMetrics["Validation F1"] = float(bestValF1)
    modelMetrics["Epochs Trained"] = 1
    modelMetrics["Feature Set"] = "ECG-only primary benchmark"
    modelMetrics["Model Artifact"] = ""
    return modelMetrics, predictionDf, countDf, {"reservoir_dim": 160, "state_features": int(xTrainReservoir.shape[1])}

advancedModelRunners = [
    (
        "Transfer Learning (In-domain DAE to Classifier)",
        "Transfer Learning / Advanced",
        lambda: trainTransferLearningModel(
            modelName="Transfer Learning (In-domain DAE to Classifier)",
            xTrain=xTrainSeq,
            yTrain=yTrainSeq,
            xVal=xValSeq,
            yVal=yValSeq,
            xTest=xTestSeq,
            yTest=yTestSeq,
            testPatientIds=testIds,
            pretrainEpochs=maxPretrainEpochs,
            finetuneEpochs=maxTrainingEpochs,
            batchSize=16,
        ),
    ),
    (
        "VICReg (Self-supervised)",
        "Transfer Learning / Advanced",
        lambda: trainVicRegModel(
            modelName="VICReg (Self-supervised)",
            xTrain=xTrainSeq,
            yTrain=yTrainSeq,
            xVal=xValSeq,
            yVal=yValSeq,
            xTest=xTestSeq,
            yTest=yTestSeq,
            testPatientIds=testIds,
            pretrainEpochs=maxPretrainEpochs,
            finetuneEpochs=maxTrainingEpochs,
            batchSize=16,
        ),
    ),
    (
        "Echo State Network (ESN)",
        "Transfer Learning / Advanced",
        lambda: trainEsnModel(
            modelName="Echo State Network (ESN)",
            xTrain=xTrainSeq,
            yTrain=yTrainSeq,
            xVal=xValSeq,
            yVal=yValSeq,
            xTest=xTestSeq,
            yTest=yTestSeq,
            testPatientIds=testIds,
        ),
    ),
]

for modelName, familyName, runner in advancedModelRunners:
    print(f"\nTraining {modelName} ...")
    try:
        modelMetrics, predictionDf, countDf, historyDict = runner()
        benchmarkResults.append(modelMetrics)
        benchmarkCountReports.append(countDf)
        benchmarkPredictionTables[modelMetrics["Model"]] = predictionDf
        benchmarkHistories[modelMetrics["Model"]] = historyDict
        print(
            f"{modelMetrics['Model']} | AUC={modelMetrics['AUC']:.4f} | F1={modelMetrics['F1']:.4f} | "
            f"Sensitivity={modelMetrics['Sensitivity']:.4f} | False Negatives={modelMetrics['False Negatives']}"
        )
        print("Prediction CSV:", modelMetrics["Prediction CSV"])
    except Exception as exc:
        recordTrainingFailure(modelName, familyName, exc)

print("\nTotal evaluated models:", len(benchmarkResults))
