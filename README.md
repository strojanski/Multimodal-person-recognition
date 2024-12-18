# Multimodal Person Recognition

## Repository
Source code is under src/. 


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

