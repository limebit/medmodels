"""Generator for MTGAN.

This module contains the Generator class, which is used to generate synthetic data
using the MTGAN model."""

from typing import Optional, Tuple, Union

import numpy as np
import sparse
import torch

from medmodels.data_synthesis.mtgan.model.generator.generator_layers import (
    GRU,
    SmoothCondition,
)
from medmodels.data_synthesis.mtgan.model.masks import sequence_mask


class Generator(torch.nn.Module):
    """Generator with GRU.

    :param torch.nn.Module: parent class.
    :type torch.nn.Module: torch.Module
    """

    def __init__(
        self,
        number_codes: int,
        max_number_admissions: int,
        hidden_dimension: int,
        attention_dimension: int,
        device: Optional[torch.device] = None,
    ) -> None:
        """Constructor for Generator of MTGAN.

        Args:
            number_codes (int): number of codes in the dataset
            max_number_admissions (int): maximum number of admissions
            hidden_dimension (int): hidden dimension of Generator
            attention_dimension (int): attention dimension of Generator
            device (torch.device): device to run the model
        """
        super().__init__()
        self.number_codes = number_codes
        self.max_number_admissions = max_number_admissions
        self.hidden_dimension = hidden_dimension
        self.attention_dimension = attention_dimension
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.gru = GRU(
            number_codes, hidden_dimension, self.max_number_admissions, device
        )
        self.smooth_condition = SmoothCondition(number_codes, attention_dimension)

    def forward(
        self,
        target_codes: torch.Tensor,
        number_admissions: torch.Tensor,
        noise: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run GRU and SmoothCondition to get the samples output.

        Args:
            target_codes (torch.Tensor): target codes from concepts group
            number_admissions (torch.Tensor): number of admissions for each patient
            noise (torch.Tensor): noise for generating

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: samples and hidden states of the GRU.
        """
        samples, hiddens = self.gru(noise)
        samples = self.smooth_condition(samples, number_admissions, target_codes)
        return samples, hiddens

    def sample(
        self,
        target_codes: torch.Tensor,
        number_admissions: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate samples with respect to the target codes.

        Args:
            target_codes (torch.Tensor): target codes from concepts group
            number_admissions (torch.Tensor): number of admissions for each patient
            noise (Optional[torch.Tensor], optional): noise for generating. Defaults to None.
            return_hiddens (bool, optional): return hidden states. Defaults to False.

        Returns:
            torch.Tensor: samples
        """
        if isinstance(target_codes, int):
            num_patients = 1
        else:
            num_patients = len(target_codes)

        if noise is None:
            noise = self.get_noise(num_patients)
        with torch.no_grad():
            mask = sequence_mask(
                number_admissions, self.max_number_admissions
            ).unsqueeze(dim=-1)
            prob, _ = self.forward(target_codes, number_admissions, noise)
            samples = torch.bernoulli(prob).to(prob.dtype)  # Make binary
            samples *= mask
            return samples

    def sample_with_hidden_states(
        self,
        target_codes: torch.Tensor,
        number_admissions: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate samples with respect to the target codes and return hidden states along with the samples.

        Args:
            target_codes (torch.Tensor): target codes from concepts group
            number_admissions (torch.Tensor): number of admissions for each patient
            noise (Optional[torch.Tensor], optional): noise for generating. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: samples and hidden states of the GRU.
        """
        if isinstance(target_codes, int):
            num_patients = 1
        else:
            num_patients = len(target_codes)

        if noise is None:
            noise = self.get_noise(num_patients)
        with torch.no_grad():
            mask = sequence_mask(
                number_admissions, self.max_number_admissions
            ).unsqueeze(dim=-1)
            prob, hiddens = self.forward(target_codes, number_admissions, noise)
            samples = torch.bernoulli(prob).to(prob.dtype)  # Make binary
            samples *= mask
            return (samples, hiddens)

    def get_noise(self, batch_size: int) -> torch.Tensor:
        """Get noise as input for generating next event.

        Args:
            batch_size (int): size of input features

        Returns:
            torch.Tensor: noise
        """
        return torch.randn(batch_size, self.hidden_dimension).to(self.device)

    def get_target_codes(self, batch_size: int) -> torch.Tensor:
        """Sample target codes.

        Args:
            batch_size (int): size of input features

        Returns:
            torch.Tensor: target codes
        """
        # TODO: why randomly through the codes and not going with the data step by step
        return torch.randint(low=0, high=self.number_codes, size=(batch_size,)).to(
            self.device
        )

    def generate_(
        self,
        number_patients: int,
        windows_distribution: torch.Tensor,
        batch_size: int,
        noise: Optional[torch.Tensor] = None,
    ) -> sparse.COO:
        """Generate synthetic data in sparse boolean matrix format.

        Args:
            number_patients (int): number of patients
            windows_distribution (torch.Tensor): distribution of the number of time windows
            batch_size (int): batch size
            noise (Optional[torch.Tensor], optional): noise for generating. Defaults to None.
        """
        synth_x = []
        for i in range(0, number_patients, batch_size):
            batch_num = min(batch_size, number_patients - i)
            if noise is not None:
                noise_i = noise[i : i + batch_num]
            else:
                noise_i = None
            target_codes = self.get_target_codes(batch_num)
            number_admissions = torch.multinomial(
                windows_distribution, num_samples=batch_num, replacement=True
            ).to(self.device)
            data = (
                self.sample(target_codes, number_admissions, noise_i)
                .cpu()
                .numpy()
                .astype(bool)
            )

            # Make it sparse
            data = sparse.COO.from_numpy(data)
            synth_x.append(data)

        # Concatenate all the data
        return sparse.concatenate(synth_x, axis=0).astype(bool)

    def get_required_number(
        self,
        windows_distribution: torch.Tensor,
        batch_size: int,
        upper_bound: int = int(1e7),
    ) -> Union[bool, int]:
        """Get the required number of samples to generate all diseases in the synthetic
        electronic health records (EHR) data.

        Args:
            windows_distribution (torch.Tensor): The distribution of the number of time
                windows for each patient.
            batch_size (int): The batch size used for generating the EHR data.
            upper_bound (int, optional): The upper bound of the number of samples to
                generate. Defaults to int(1e7).

        Returns:
            Union[bool, int]: The required number of samples to generate all diseases in
                the synthetic EHR data if it is less than the upper bound, False otherwise.
        """
        code_types = torch.zeros(
            self.number_codes, dtype=torch.bool, device=self.device
        )
        required_number = 0
        while True:
            number = np.random.randint(
                low=np.floor(0.5 * batch_size), high=np.floor(1.5 * batch_size)
            )
            required_number += number

            # Generate n samples
            number_admissions = (
                torch.multinomial(
                    windows_distribution, num_samples=number, replacement=True
                )
                + 1
            ).to(self.device)
            target_codes = self.get_target_codes(number)
            data = self.sample(
                target_codes=target_codes,
                number_admissions=number_admissions,
            )
            code_types = torch.logical_or(code_types, data.sum(dim=1).sum(dim=0) > 0)
            total_code_types = code_types.sum()

            if total_code_types == self.max_number_admissions:
                return required_number
            if required_number >= upper_bound:
                return False  # Cannot generate all diseases under the upper bound
