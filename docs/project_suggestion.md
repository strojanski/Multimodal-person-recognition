# Research track project suggestion

### Task Description
Modality fusion for biometric recognition - Fusing modalities and comparing performance based on fusion type / fused modalities.
Suggested modalities: Iris, Face, Gait 

Suggested datasets: 
1. Create synthetic data from merging separate datasets, eg. CASIA-B (Gait), CelebA (Faces) and Casia-Iris-Thousand (Iris) from the set of available datasets from the table below.
2. Find a multimodal dataset, however not many with sufficient data

Found datasets:

| Dataset               | # Subjects    | Modality      | # Samples     |
| ---------             | ------------- | ---------     | ---------     |
| Casia-Iris-Thousand   | 1000          | Iris          | 20 000        |
| Casia Palmprint       | 312           | Palmprint     | 5502          |
| Casia-FaceV5          | 500           | Face          | 2500          |
| CelebFaces            | 10 177        | Face          | 202 599       |
| Casia Fingerprint 5   | 500           | Fingerprint   | 20 000        |

#### Experiments
1. Create a new dataset from combination of a subset of datasets from table
2. Find a feature extraction model for each modality (eg. FaceNet for face)
3. Pick a fusion model (EmbraceNet)
4. Evaluate performance with different combinations of modalities (for recognition).


### Goal
Test how different combinations of modalities improve mulitmodal recognition (up to 3-4 modalities). Test EmbraceNet fusion techniques. Determine a more specific goal due to time constraint. 



[Embracenet](https://arxiv.org/pdf/1904.09078v1)
