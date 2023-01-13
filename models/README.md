# Emotional model

In this step, multiple SVM-based emotional models were constructed and evaluated by performance metrics. A sensitivity analysis was computed while variating the kernel, gamma, degree, and C parameters of the SVM with a 5-fold CV. Also, it was elaborated different emotional classifications take into account different data, with or without an ensemble, and with different classes.

The models were evaluated by the following performance metrics:
-	Balanced Accuracy (BA);
-	Cohen’s Kappa Score (CKS)
-	Matthews Correlation Coefficient (MCC);
-	Area Under the Curve of the Receiver Operating Characteristics (ROC-AUC).

The optimal parameters are chosen by the maximum value of BA, conjugated with the minimum difference in BA between the train and test set.

The implementation is prepared for classification with two or three classes within an emotion.

