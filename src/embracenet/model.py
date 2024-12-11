import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(".."))
from embracenet import EmbraceNet


class TrimodalModel:
    def __init__(self):
        self.lr = 1e-3
        self.model_dropout = True

    def prepare(self, is_training, input_size_list, global_step=0, n_classes=500):
        # config. parameters
        """
        Initialize the model and related parameters.

        Args:
            is_training (bool): True if it is in training mode.
            global_step (int, optional): The current global step. Defaults to 0.
        """
        self.global_step = global_step
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # PyTorch model
        self.model = EmbraceNetBimodalModule(
            device=self.device,
            is_training=is_training,
            input_size_list=input_size_list,
            n_classes=n_classes,
            global_step=global_step,
        )
        if is_training:
            self.optim = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr
            )
            self.loss_fn = nn.NLLLoss()

        # configure device
        self.model = self.model.to(self.device)

    def save(self, base_path):
        save_path = os.path.join(base_path, "model_%d.pth" % (self.global_step))
        torch.save(self.model.state_dict(), save_path)

    def restore(self, ckpt_path):
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))

    def get_model(self):
        return self.model

    def train_step(self, input_list, truth_list, summary=None):
        """
        Perform a training step with a batch of inputs and ground truth.

        Args:
            input_list (list): List of batches, each containing 2 modalities with different shapes.
            truth_list (list): Ground truth labels for each example in the batch.
            summary (SummaryWriter, optional): If not None, write loss and learning rate to the summary.

        Returns:
            float: The loss of the current step.
        """
        # Initialize list for modality tensors
        batch_size = len(input_list)
        modality_1_tensors = []
        modality_2_tensors = []
        modality_3_tensors = []

        # Convert each modality per example in the batch to tensors
        for i in range(batch_size):
            modality_1 = torch.as_tensor(
                input_list[i][0], dtype=torch.float, device=self.device
            )  # shape [1, 60]
            modality_2 = torch.as_tensor(
                input_list[i][1], dtype=torch.float, device=self.device
            )  # shape [1, 43]
            modality_3 = torch.as_tensor(
                input_list[i][2], dtype=torch.float, device=self.device
            )
            
            modality_1_tensors.append(modality_1)
            modality_2_tensors.append(modality_2)
            modality_3_tensors.append(modality_3)

        # Stack tensors to create two modality tensors of shape [batch_size, ...]
        modality_1_tensor = torch.cat(modality_1_tensors, dim=0)
        modality_2_tensor = torch.cat(modality_2_tensors, dim=0)
        modality_3_tensor = torch.cat(modality_3_tensors, dim=0)

        # Combine modality tensors into a list for model input
        input_tensors = [modality_1_tensor, modality_2_tensor, modality_3_tensor]
        truth_tensor = torch.as_tensor(truth_list, dtype=torch.long, device=self.device)

        # Forward pass through the model
        output_tensor = self.model(input_tensors)  # Pass modality list to model

        loss = self.loss_fn(output_tensor, truth_tensor)
        print(loss)

        # Adjust learning rate
        lr = self.lr
        for param_group in self.optim.param_groups:
            param_group["lr"] = lr

        # Backpropagation
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # Increment global step
        self.global_step += 1

        # Write summary
        if summary is not None:
            summary.add_scalar("loss", loss, self.global_step)
            summary.add_scalar("lr", lr, self.global_step)

        return loss.item()

    def predict(self, input_list):
        # numpy to torch
        """
        Predicts the output probabilities and classes for a batch of input data.

        Args:
            input_list (list): A list of two numpy arrays, each of shape
                [batch_size, 1, 28, 14].

        Returns:
            tuple: A tuple containing the output probabilities and classes.
                The first element of the tuple is a numpy array of shape
                [batch_size, 10] representing the output probabilities. The
                second element of the tuple is a numpy array of shape
                [batch_size] representing the output classes.
        """
        input_tensor = torch.tensor(input_list, dtype=torch.float32, device=self.device)

        # get output
        output_tensor = self.model(input_tensor)

        # finalize
        class_list = output_tensor.argmax(dim=-1).detach().cpu().numpy()
        prob_list = output_tensor.detach().cpu().numpy()

        # finalize
        return prob_list, class_list


import torch.nn.init as init


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)  # Xavier initialization for Linear layers
        if m.bias is not None:
            init.constant_(m.bias, 0)  # Initialize biases to zero
    elif isinstance(m, nn.Conv2d):
        init.kaiming_normal_(
            m.weight, mode="fan_out", nonlinearity="relu"
        )  # Kaiming init for Convolutional layers
        if m.bias is not None:
            init.constant_(m.bias, 0)


class EmbraceNetBimodalModule(nn.Module):
    def __init__(
        self, device, is_training, input_size_list, n_classes, global_step=0
    ):
        """
        Initialize an EmbraceNetBimodalModule.

        Args:
            device: A "torch.device()" object to allocate internal parameters of the EmbraceNetBimodalModule.
            is_training: A boolean indicating whether the module is in the training phase or not.
            input_size_list: A list of two integers, each of which is the size of the respective input data.
        """
        super(EmbraceNetBimodalModule, self).__init__()

        self.model_dropout = True  # True
        # input parameters
        self.device = device
        self.is_training = is_training

        self.input_size_list = input_size_list
        self.n_classes = n_classes

        # embracenet
        self.embracenet = EmbraceNet(
            device=self.device,
            input_size_list=self.input_size_list,
            embracement_size=256,
            bypass_docking=False,
        )

        # Initialize random weights
        if global_step == 0:
            self.embracenet.apply(initialize_weights)

        # post embracement layers
        self.post = nn.Linear(in_features=256, out_features=n_classes)

    def forward(self, x):
        # separate x into left/right

        for name, param in self.embracenet.named_parameters():
            assert not torch.isnan(param).any(), f"NaN in parameter {name}"
            assert not torch.isinf(param).any(), f"Inf in parameter {name}"


        x_1 = x[0]
        x_1 = x_1.view(32, -1) #self.input_size_list[0])

        x_2 = x[1]
        x_2 = x_2.view(32, -1) # self.input_size_list[1])

        x_3 = x[2]
        x_3 = x_3.view(32, -1) # self.input_size_list[2])
        


        # dropout during training, shape 32
        availabilities = None
        if self.is_training and self.model_dropout:
            dropout_prob = torch.rand(1, device=self.device)[0]
            if dropout_prob >= 0.5:
                target_modalities = torch.round(
                    torch.rand([32], device=self.device)
                ).to(torch.int64)
                availabilities = nn.functional.one_hot(
                    target_modalities, num_classes=3
                ).float()

        # embrace
        x_embrace = self.embracenet([x_1, x_2, x_3], availabilities=availabilities)

        # return x_embrace
        # employ final layers
        x = self.post(x_embrace)

        # output softmax
        return nn.functional.log_softmax(x, dim=-1)
