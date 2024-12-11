The table below displays results of different modality combinations. The results were obtained by training a Random Forest Classifier with 10 estimators (~2min time to train). 

Features were extracted by using 50% of each class (subject) data to train the model and the rest was turned into embeddings. When using RF for classification we used a train_test_split of 80:20.

To improve results we can try out other models (XGB) and dimensionality reduction on embeddings (top 10 features by PCA) and redo. 


| Modality              | Accuracy  | F1        | Model             |
|----------------       |---------- |-------    |----------------   |
| Face                  | 0.225     | 0.206     | RF                |
| Iris                  | 0.738     | 0.703     | RF                |
| Fingerprint           | 0.688     | 0.665     | RF                |
| Face+Iris             | 0.996     | 0.957     | RF                |
| Face+Fingerprint      | 0.883     | 0.872     | RF                |
| Iris + Fingerprint    | 0.974     | 0.972     | RF                |
| All                   | 0.979     | 0.978     | RF                |

We see huge improvements when using combinations. 

The combinations were made by taking 2 modalities and oversampling the one with less samples and then concatenating them. 

The following table displays scores achieved on embeddings reduced to 100 components with RF with 50 estimators.
Face kept 99% explained variance, iris 88 and fingerprints 89.

| Modality              | Accuracy  | F1        | Model             |
|----------------       |---------- |-------    |----------------   |
| Face                  | 0.361     | 0.316     | RF                |
| Iris                  | 0.893     | 0.873     | RF                |
| Fingerprint           | 0.842     | 0.825     | RF                |
| Face+Iris             | 0.995     | 0.994     | RF                |
| Face+Fingerprint      | 0.951     | 0.942     | RF                |
| Iris + Fingerprint    | 0.995     | 0.993     | RF                |
| All                   | 0.997     | 0.997     | RF                |

To further test the robustness of this approach we could rerun the experiments with subsampled data, since the oversampling introduces some degree of overfitting.


The following table displays scores of subsampled original embeddings, again RF with 10 estimators.

| Modality              | Accuracy  | F1        | Model             |
|----------------       |---------- |-------    |----------------   |
| Face                  | 0.225     | 0.206     | RF                |
| Iris                  | 0.738     | 0.703     | RF                |
| Fingerprint           | 0.688     | 0.665     | RF                |
| Face+Iris             | 0.810     | 0.768     | RF                |
| Face+Fingerprint      | 0.823     | 0.796     | RF                |
| Iris + Fingerprint    | 0.861     | 0.840     | RF                |
| All                   | 0.865     | 0.834     | RF                |

In this case the improvements are less extreme but still obvious, as every combination performs better than a single modality by a margin of about 10%.

The last table shows results of subsampling + PCA with 50 estimators:

| Modality              | Accuracy  | F1        | Model             |
|----------------       |---------- |-------    |----------------   |
| Face                  | 0.361     | 0.316     | RF                |
| Iris                  | 0.893     | 0.873     | RF                |
| Fingerprint           | 0.842     | 0.825     | RF                |
| Face+Iris             | 0.950     | 0.937     | RF                |
| Face+Fingerprint      | 0.926     | 0.912     | RF                |
| Iris + Fingerprint    | 0.949     | 0.940     | RF                |
| All                   | 0.975     | 0.968     | RF                |
