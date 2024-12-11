import copy

import numpy as np

# data loader for bimodal MNIST and Fashion-MNIST


class TrimodalLoader:
    def __init__(self, iris, fingerprint, face, y):
        self.iris = iris
        self.fingerprint = fingerprint
        self.face = face
        self.y = y
        
        self.data_list = []
        self.prepare()

        super().__init__()

    def prepare(self):
        """
        Used to read data from disk, fills self.data_list with ({"left": mod1, "right": mod2, "label": label)
        """
        for i, fi, fa, l in zip(self.iris, self.fingerprint, self.face, self.y):
            self.data_list.append({"iris": i, "fingerprint": fi, "face": fa, "label": l})

    def get_num_data(self):
        return len(self.data_list)

    def get_batch(self, batch_size):
        """
        Retrieves a batch of randomly selected data from the dataset.

        Args:
            batch_size (int): The number of data points to retrieve in the batch.

        Returns:
            tuple: A tuple containing the input data (list of lists of modalities) and the labels.
        """
        input_list = []
        truth_list = []

        for _ in range(batch_size):
            input_data, label = self.get_random_data_pair()
            input_list.append(input_data)
            truth_list.append(label)

        return input_list, truth_list  # [[mod1, mod2, ...]], label

    def get_random_data_pair(self):
        """
        Randomly selects and returns a data pair from the dataset.

        Returns:
            tuple: A tuple containing the input data (list of modalities) and the label.
        """
        data_index = np.random.randint(self.get_num_data())

        # retrieve data
        input_data, label, _ = self.get_data_pair(
            data_index=data_index
        )  # [mod1, mod2, ...], label

        # finalize
        return input_data, label

    def get_data_pair(self, data_index):
        input_data, label = self._get_input_data(index=data_index)
        # finalize
        return input_data, label, data_index

    def _get_input_data(self, index):
        """
        Fetches the input data and label for a given index.
        
        Args:
            index (int): Index of the data to fetch.
        
        Returns:
            tuple: ([mod1, mod2, mod3], label) where mods are the modalities (iris, fingerprint, face).
        """
        # Retrieve the data point from the list
        data = self.data_list[index]  # {'iris': ..., 'fingerprint': ..., 'face': ..., 'label': ...}
        
        # Combine modalities into a single list
        input_data = [
            np.array(data["iris"]),        # Convert tensor to NumPy array
            np.array(data["fingerprint"]), # Convert tensor to NumPy array
            np.array(data["face"])         # Convert tensor to NumPy array
        ]
        
        # Copy the label as a NumPy array
        label = np.array(data["label"])
        
        return input_data, label



    # def _get_input_data(self, index):
    #     data = self.data_list[
    #         index
    #     ]  # dict {"left": image_left, "right": image_right, "label": label}
    #     return copy.deepcopy(
    #         [[data["iris"]], [data["fingerprint"]] [data["face"]]]
    #     ), copy.deepcopy(
    #         data["label"]
    #     )  # [mod1, mod2, ...], label
