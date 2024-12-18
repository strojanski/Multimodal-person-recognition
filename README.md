# Multimodal Person Recognition

## Repository
Source code is under `src/` and is divided into folders corresponding to modality feature extraction, concatenation and embracenet classification.
In the `docs/` folder you can find some overview of the modailty fusion techniques. 



## Modality fusion 

### Datasets
| Dataset               | Subjects | Modality    | Samples   |
|------------------------|----------|-------------|-----------|
| CelebFaces            | 10,177   | Face        | 202,599   |
| Casia-Iris-Thousand   | 1,000    | Iris        | 20,000    |
| Casia Fingerprint v5  | 500      | Fingerprint | 20,000    |



### Feature extraction models

CelebFaces -> FaceNet
Iris -> EfficientNet
Fingerprint -> EfficientNet

### Fusion
EmbraceNet, concatenation

### Classification
Transformers, MLP, ...



### Notes
Copy of any published work has to be forwarded to casia professor.

