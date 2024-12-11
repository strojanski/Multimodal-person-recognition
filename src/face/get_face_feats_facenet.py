import os

import numpy as np
import pandas as pd
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

mtcnn = None
resnet = None


def init():
    global mtcnn
    global resnet

    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(image_size=100)

    # Create an inception resnet (in eval mode):
    # resnet = InceptionResnetV1(pretrained="vggface2").eval()
    resnet = InceptionResnetV1(pretrained="casia-webface").eval()


def get_embeddings(imgs):
    global resnet
    embeddings = []
    invalid_labels = []
    for i, img in enumerate(imgs):
        try:
            embedding = resnet(img.unsqueeze(0))
            embeddings.append(embedding.detach().numpy())
            if i % 100 == 0:
                print(i)
        except BaseException:
            invalid_labels.append(i)
            continue
    return embeddings, invalid_labels



def preprocess(root_path):
    global mtcnn

    preprocessed = []
    labels = []
    for sub in os.listdir(root_path):
        print(sub)
        files = os.listdir(f"{root_path}/{sub}/face/")
        # random.shuffle(list(files))
        # if sub == "s002":
        # break

        for img in files:
            # 2 fps
            # if int(img.split("_")[2].split(".")[0]) % 5 != 0:
            # continue

            i = Image.open(f"{root_path}/{sub}/face/{img}")
            i = mtcnn(i)
            preprocessed.append(i)
            labels.append(sub)
    print(len(preprocessed))
    return preprocessed, labels


def test_embeddings(embeddings, labels):
    y_pred = []
    y_true = labels

    print(len(labels), len(embeddings))

    for i in range(len(embeddings)):
        distances = np.zeros(len(embeddings))
        for j in range(len(embeddings)):
            if i != j:
                distances[j] = np.linalg.norm(embeddings[i] - embeddings[j])
            else:
                distances[j] = np.inf

        min_ix = np.argmin(distances)
        # print(np.min(distances), np.max(distances))

        y_pred.append(labels[min_ix])

    y_true = np.array(y_true)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    return accuracy, f1


def embeddings_to_df(embeddings, labels):
    dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
    return pd.DataFrame(dists, columns=labels, index=labels)


def pca_clustering(embeddings, labels):
    # Convert embeddings and labels to numpy arrays
    embeddings = np.array(embeddings)
    labels = np.array(labels)

    # Perform PCA to reduce the dimensionality to 2D
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)

    # Encode labels to integers for coloring
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Compute pairwise distances between embeddings
    distances = np.linalg.norm(
        embeddings[:, np.newaxis] - embeddings[np.newaxis, :], axis=2
    )

    # Set the diagonal to infinity to avoid zero distance to self
    np.fill_diagonal(distances, np.inf)

    # Get the indices of the nearest neighbors
    nearest_neighbors = np.argmin(distances, axis=1)

    # Get the predicted labels based on the nearest neighbor
    y_pred = labels[nearest_neighbors]

    # Calculate accuracy and F1 score
    accuracy = accuracy_score(labels, y_pred)
    f1 = f1_score(labels, y_pred, average="macro")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return pca_result, accuracy, f1


if __name__ == "__main__":
    init()
    root_path = "../../data/combined_dataset"
    print("Preprocessing")
    preprocessed, labels = preprocess(root_path)
    print("Getting embeddings")
    embeddings, invalid_labels = get_embeddings(preprocessed)
    preprocessed = []

    print(f"{len(invalid_labels)} unsuccessful embeddings")
    labels = [item for idx, item in enumerate(labels) if idx not in invalid_labels]
    embeddings = [
        item for idx, item in enumerate(embeddings) if idx not in invalid_labels
    ]

    # Might skew the score a bit but above code doesn't work
    if len(labels) > len(embeddings):
        labels = labels[: len(embeddings)]

    assert len(labels) == len(embeddings)
    print("n_classes: ", len(set(labels)))
    # Code for creating a dataframe of distances
    # df = test_2(embeddings, labels)
    # print(df)
    # df.to_csv("embeddings.csv")
    np_emb = []
    cnt = 0
    for emb, lab in zip(embeddings, labels):
        # if type(emb) == torch.Tensor():
        # if emb.is_cuda:
        # emb = emb.cpu()

        # emb = emb.numpy()
        emb = emb.squeeze(0)
        np_emb.append(emb)
        print(emb.shape)
        np.save(f"face_embeddings/{lab}_{cnt}.npy", emb)
        cnt += 1
    # exit()
