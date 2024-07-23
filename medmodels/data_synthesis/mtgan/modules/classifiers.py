from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Union

import numpy as np
import sparse
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.optim.adam import Adam

from medmodels.data_synthesis.mtgan.model.loaders import MTGANDataLoader
from medmodels.data_synthesis.mtgan.model.masks import sequence_mask
from medmodels.medrecord.types import MedRecordValue, NodeIndexInputList

if TYPE_CHECKING:
    from medmodels.data_synthesis.mtgan.modules.postprocessor import (
        PostprocessingHyperparameters,
    )


class TimeLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hyperparameters: PostprocessingHyperparameters,
        device: torch.device,
        output_size: int = 1,
    ):
        super(TimeLSTM, self).__init__()
        self.hidden_dim = hyperparameters["hidden_dim"]
        self.number_layers = hyperparameters["number_layers"]
        self.learning_rate = hyperparameters["learning_rate"]
        self.epochs = hyperparameters["training_epochs"]
        self.device = device

        self.lstm = nn.LSTM(
            input_size, self.hidden_dim, self.number_layers, batch_first=True
        )
        self.fc = nn.Linear(self.hidden_dim, output_size)
        self.to(self.device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        h0 = torch.zeros(self.number_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.number_layers, x.size(0), self.hidden_dim).to(x.device)

        # here the LSTM takes directly x as input
        out, _ = self.lstm(x, (h0, c0))
        out = out * mask.unsqueeze(2)

        out = self.fc(out)

        return out.view(-1)

    def train_model(self, train_loader: MTGANDataLoader):
        """
        Train the model.

        Args:
            train_loader (MTGANDataLoader): The data loader with the training data.
        """
        criterion = nn.MSELoss()
        optimizer = Adam(self.parameters(), lr=self.learning_rate)

        for _ in range(self.epochs):
            for _, (x, number_admissions, y) in enumerate(train_loader):
                x = x.to(self.device)
                number_admissions = number_admissions.to(self.device)
                y = y.to(self.device)

                mask = sequence_mask(number_admissions, x.size(1)).to(x.device)
                outputs = self.forward(x, mask)

                padded_y = torch.zeros_like(outputs)
                start = 0
                for idx, num_adm in enumerate(number_admissions):
                    start_padded = idx * x.size(1)
                    end = start + num_adm
                    padded_y[start_padded : start_padded + num_adm] = y[start:end]
                    start = end
                loss = criterion(outputs * mask.view(-1), padded_y)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

    def evaluate(self, test_loader: MTGANDataLoader):
        """Evaluates the model.

        Args:
            test_loader (MTGANDataLoader): The data loader with the test data.

        Returns:
            torch.Tensor: The predicted output.
        """
        criterion = nn.MSELoss()
        with torch.no_grad():
            cumulative_test_loss = 0
            for _, (x_test, number_admissions_test, y_test) in enumerate(test_loader):
                self.eval()  # switch to evaluation mode
                x_test = x_test.to(self.device)
                number_admissions_test = number_admissions_test.to(self.device)
                y_test = y_test.to(self.device)

                mask_test = sequence_mask(number_admissions_test, x_test.size(1)).to(
                    self.device
                )
                outputs_test = self.forward(x_test, mask_test)
                padded_y = torch.zeros_like(outputs_test)
                start = 0
                for idx, num_adm in enumerate(number_admissions_test):
                    start_padded = idx * x_test.size(1)
                    end = start + num_adm
                    padded_y[start_padded : start_padded + num_adm] = y_test[start:end]
                    start = end
                loss_test = criterion(outputs_test * mask_test.view(-1), padded_y)

                cumulative_test_loss += loss_test.item()

            avg_test_loss = cumulative_test_loss / len(test_loader)
            return avg_test_loss

    def predict(self, synthetic_data: sparse.COO) -> NDArray[np.float32]:
        """Predicts the output for synthetic data.

        Args:
            synthetic_data (sparse.COO): The synthetic data.

        Returns:
            torch.Tensor: The predicted output. The first column is always 0.

        Raises:
            ValueError: If the absolute time windows are not in ascending order.
        """
        self.eval()
        synthetic_tensor = (
            torch.from_numpy(synthetic_data.todense()).float().to(self.device)
        )
        number_admissions = get_admission_lengths(synthetic_data).to(self.device)
        with torch.no_grad():
            absolute_time_windows = []
            mask = sequence_mask(number_admissions, synthetic_tensor.size(1)).to(
                synthetic_tensor.device
            )
            absolute_time_windows = self(synthetic_tensor, mask).view(
                -1, synthetic_tensor.size(1)
            )

        absolute_time_windows = absolute_time_windows.cpu().numpy()
        absolute_time_windows[:, 0] = 0

        # Make sure there is an ascending order for the absolute time windows in each row
        differences = np.diff(absolute_time_windows, axis=1) * mask[:, 1:].numpy()
        if np.any(differences < 0):
            raise (ValueError("The absolute time windows are not in ascending order."))

        return absolute_time_windows


class AttributeRegression(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hyperparameters: PostprocessingHyperparameters,
        device: torch.device,
    ):
        super(AttributeRegression, self).__init__()
        self.hyperparameters = hyperparameters
        self.device = device
        self.type_model = "regression"
        self.to(device)

        self.lstm = nn.LSTM(input_size, hyperparameters["hidden_dim"], batch_first=True)
        self.fc = nn.Linear(hyperparameters["hidden_dim"], output_size)
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out * mask

        return self.fc(lstm_out)


class AttributeCategorical(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_categories: int,
        output_size: int,
        hyperparameters: PostprocessingHyperparameters,
        device: torch.device,
    ):
        super(AttributeCategorical, self).__init__()
        self.hyperparameters = hyperparameters
        self.num_categories = num_categories
        self.output_size = output_size
        self.device = device
        self.type_model = "categorical"
        self.to(device)

        self.lstm = nn.LSTM(input_size, hyperparameters["hidden_dim"], batch_first=True)
        self.fc = nn.Linear(hyperparameters["hidden_dim"], num_categories * output_size)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out * mask
        logits = self.fc(lstm_out)
        return logits.view(
            logits.size(0), logits.size(1), self.output_size, self.num_categories
        )


class AttributeClassifier(nn.Module):
    def __init__(self, model: Union[AttributeRegression, AttributeCategorical]):
        super(AttributeClassifier, self).__init__()
        self.model = model
        self.type_model = model.type_model
        self.epochs = model.hyperparameters["training_epochs"]
        self.lr = model.hyperparameters["learning_rate"]
        self.device = model.device

    def train_model(self, train_loader: MTGANDataLoader):
        optimizer = Adam(self.model.parameters(), lr=self.lr)

        for _ in range(self.epochs):
            for x, number_admissions, y in train_loader:
                x = x.to(self.device)
                number_admissions = number_admissions.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                mask = sequence_mask(number_admissions, x.size(1)).to(x.device)
                output = self.model(x, mask)

                mask_y = torch.isnan(y)
                output[mask_y] = np.nan
                loss = self.model.criterion(output * mask_y, y)

                loss.backward()
                optimizer.step()

    def predict(
        self,
        synthetic_data: sparse.COO,
        concepts: Union[List[MedRecordValue], NodeIndexInputList],
    ) -> List[Any]:
        mask = sequence_mask(
            get_admission_lengths(synthetic_data), synthetic_data.shape[1]
        ).to(self.model.device)
        synthetic_tensor = (
            torch.from_numpy(synthetic_data.todense()).float().to(self.model.device)
        )
        with torch.no_grad():
            predictions = self.model(synthetic_tensor, mask)
            mask_output = torch.isnan(predictions[:, :, concepts])
            predictions[mask_output] = None
            if self.type_model == "categorical":
                return predictions.cpu().argmax(dim=3).tolist()
            return predictions.cpu().tolist()


def get_admission_lengths(data: sparse.COO) -> torch.Tensor:
    """Gest the number of admissions (up to the last admission with at least one code) for each patient.

    Args:
        data(sparse.COO): 3D boolean matrix of shape (num_patients, max_admission_num,
            code_num), 1 if the code is present, 0 otherwise.

    Returns:
        torch.Tensor: The number of admissions for each patient.
    """
    admissions_with_codes = data.sum(axis=2).todense()
    number_admissions = np.zeros(data.shape[0], dtype=int)

    for i in range(data.shape[0]):
        non_empty_admissions = np.where(admissions_with_codes[i, :] > 0)[0]
        if non_empty_admissions.size > 0:
            number_admissions[i] = non_empty_admissions[-1] + 1

    return torch.from_numpy(number_admissions)
