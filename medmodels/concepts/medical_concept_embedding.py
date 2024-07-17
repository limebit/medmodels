import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from time import time
from medmodels.concepts.mce_dataclass import DataClassMCE
from medmodels.medrecord.types import NodeIndex

# import logging


# TODO: Logging setup has to be changed, help needed
def log(text):
    print(text)
    # logging.info(text)


class MCEwTAA:
    def __init__(
        self,
        data: pd.DataFrame,
        embedding_size: int = 750,
        time_interval: int = 7,
        time_scope_radius: int = 50,
        max_events: int = 100,
        min_total_events: int = 5,
        seed: int = 100,
        negative_samples: int = 5,
        warm_init: dict = {},
        loss_function: str = "bce_sigmoid",
    ):
        """Medical Concept Embedding with Time Aware Attention

        :param data: Data to train on
        :type data: pd.DataFrame
        :param embedding_size: Size of the embedding vector. Defaults to 100.
        :type embedding_size: int, optional
        :param time_interval: Time interval to discretize time periods (e.g.
            time_interval=7 -> weeks, time_interval=30 -> months). Defaults to 7.
        :type time_interval: int, optional
        :param time_scope_radius:  Look in time intervals back and forward to search for
            related concepts. Defaults to 50.
        :type time_scope_radius: int, optional
        :param max_events: Limit of events to consider as context for computational
            reasons. Defaults to 60.
        :type max_events: int, optional
        :param min_total_events: Minimum number of events a code must appear to be
            considered in the training set. Defaults to None.
        :type min_total_events: int, optional
        :param negative_samples: Number of negative samples in training context.
            Defaults to 5.
        :type negative_samples: int, optional
        :param seed: Seed for random number generator. Defaults to 100.
        :type seed: int, optional
        :param warm_init: Warm start the model with a pretrained model.
            Defaults to None.
        :type warm_init: OrderedDict, optional
        :param loss_function: Loss function to use. Defaults to "bce_sigmoid". Options:
            "bce_sigmoid", "bce_softmax".
        :type loss_function: str, optional
        """

        if "params" in warm_init.keys():
            hyperparams = warm_init["hyperparams"]

            self.embedding_size = hyperparams["embedding_size"]
            self.time_interval = hyperparams["time_interval"]
            self.time_scope_radius = hyperparams["time_scope_radius"]
            self.max_events = hyperparams["max_events"]
            self.min_total_events = hyperparams["min_total_events"]
            self.negative_samples = hyperparams["negative_samples"]
            self.seed = hyperparams["seed"]
            self.loss_function = hyperparams["loss_function"]

        else:
            self.embedding_size = embedding_size
            self.time_interval = time_interval
            self.time_scope_radius = time_scope_radius
            self.max_events = max_events
            self.min_total_events = min_total_events
            self.negative_samples = negative_samples
            self.seed = seed

            assert loss_function in [
                "bce_sigmoid",
                "bce_softmax",
            ], "Invalid loss function"
            self.loss_function = loss_function

        # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initiate data class as it is required to deal with the data
        self.data = DataClassMCE(
            data=data,
            time_interval=self.time_interval,
            time_scope_radius=self.time_scope_radius,
            max_events=self.max_events,
            min_total_events=self.min_total_events,
            seed=self.seed,
        )

        # Model
        self.model = MCEwTAAModel(
            data=self.data,
            embedding_size=self.embedding_size,
            time_scope_radius=self.time_scope_radius,
            device=self.device,
            loss_function=self.loss_function,
        ).to(self.device)

        if "emb_params" in warm_init.keys():
            emb_params = warm_init["emb_params"]

            self.model.taa_bias = torch.nn.Parameter(emb_params["taa_bias"])
            self.model.taa_matrix = torch.nn.Parameter(emb_params["taa_matrix"])
            self.model.context_embedding.weight = torch.nn.Parameter(
                emb_params["context_embedding.weight"]
            )
            self.model.target_embedding.weight = torch.nn.Parameter(
                emb_params["target_embedding.weight"]
            )

    def fit(
        self,
        batch_size: int = 512,
        epochs: int = 50,
        lr: float = 3e-4,
        weight_decay: float = 1e-6,
    ) -> None:
        """Train the Medical Concept Embedding

        :param batch_size: Batch size for training interation.
            Defaults to 512.
        :type batch_size: int, optional
        :param epochs: Number of epochs to train. Defaults to 50.
        :type epochs: int, optional
        :param lr: Learning rate for training. Defaults to 3e-4.
        :type lr: float, optional
        :param weight_decay: Weight decay for training. Defaults to 1e-6.
        :type weight_decay: float, optional
        :return: None
        """
        # TODO: Logging setup has to be changed, help needed
        """
        logging.basicConfig(
            filename=os.path.join(experiment_path, "mce_log_{}.txt".format(run_time)),
            filemode="a",
            format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            level=logging.DEBUG,
        )
        """

        log(
            f"""This Computation is running on {self.device} with params:
            Time Interval: {self.time_interval}
            Negative Samples: {self.negative_samples}
            Embedding Size: {self.embedding_size}
            Time Scope: {self.time_scope_radius}
            Max Event: {self.max_events}
            Min Total Events: {self.min_total_events}
            Batch Size: {batch_size}
            Epochs: {epochs}"""
        )

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        if self.loss_function == "bce_sigmoid":
            loss_function = torch.nn.BCEWithLogitsLoss()
        elif self.loss_function == "bce_softmax":
            loss_function = torch.nn.BCELoss()
        y_pos = torch.ones((batch_size, 1))
        y_neg = torch.zeros((batch_size, self.negative_samples))
        y = torch.cat([y_pos, y_neg], dim=1)

        self.train_epoch_loss, self.validation_epoch_loss = [], []

        # Training
        for epoch in tqdm(range(epochs)):
            start = time()
            train_batch_loss, validation_batch_loss = [], []

            # TRAIN ITERATION
            for context, target in self.data.loader(end=0.8, batch_size=batch_size):
                current_batch_size = target.shape[0]

                # context contains the all events and its time point that are
                # corresponding to the current target matrix from this batch. It is a
                # matrix in shape [current_batch_size, H, 2]  where H is equivalent to
                # the highest number of context events in this batch, either limimted
                # by max_events or data. current_batch_size may differ to batch_size,
                # e.g. if it is the last batch. The last dimension is the code of the
                # context event and its point in time.

                # target is a matrix in shape [batch_size, 2]. The last dimension is
                # the code of the target event and its point in time.

                # draw false target ids to let the model discriminate between real
                # and fake ids to learn relations between context and targets

                # the position of entries in the negative sampling table.
                # will be transfered to ids in the lines
                # drawing like this has only computational reasons
                # negative_target_pos = torch.randint(
                #     len(data.expanded_ns_table),
                #     (current_batch_size, self.negative_samples),
                # )

                # the ids of [self.negative_samples] negative targes (e.g. 2),
                # meaning targets that did not happen within the given context
                negative_target_ids = self.negative_sampling(
                    self.data.expanded_negative_sampling_table,
                    current_batch_size,
                    target[:, :1],
                )

                # Target ID, negative samples IDs, Target time.
                mixed_target = torch.cat(
                    [target[:, :1], negative_target_ids, target[:, 1:]], dim=-1
                )
                # prediction and loss
                optimizer.zero_grad()

                scores = self.model(
                    context.to(self.device), mixed_target.to(self.device)
                )
                loss = loss_function(
                    scores.to(self.device), y[:current_batch_size].to(self.device)
                )
                # Compute gradients and update the embeddings.
                loss.backward()
                optimizer.step()

                train_batch_loss.append(loss.detach().item())
            self.train_epoch_loss.append(np.mean(train_batch_loss))

            # TEST ITERATION
            for context, target in self.data.loader(start=0.8, batch_size=batch_size):
                current_batch_size = target.shape[0]
                negative_target_ids = self.negative_sampling(
                    self.data.expanded_negative_sampling_table,
                    current_batch_size,
                    target[:, :1],
                )

                mixed_target = torch.cat(
                    [target[:, :1], negative_target_ids, target[:, 1:]], dim=-1
                )

                # prediction and loss
                optimizer.zero_grad()

                scores = self.model(
                    context.to(self.device), mixed_target.to(self.device)
                )
                loss = loss_function(
                    scores.to(self.device), y[:current_batch_size].to(self.device)
                )

                validation_batch_loss.append(loss.detach().item())
            self.validation_epoch_loss.append(np.mean(validation_batch_loss))

            log(
                f"\nIteration {epoch + 1}/{epochs} took "
                f"{round((time() - start) / 60, 2)} min: "
                f"Train Loss {np.mean(train_batch_loss):.5f}, "
                f"Validation Loss {np.mean(validation_batch_loss):.5f}"
            )

    def negative_sampling(
        self, expanded_ns_table: torch.Tensor, batch_size: int, targets: torch.Tensor
    ) -> torch.Tensor:
        """Create some negative ids.

        Draw false target ids to let the model discriminate between real
        and fake ids to learn relations between context and targets. The position of
        entries in the negative sampling table will be transfered to ids in the lines

        :param expanded_ns_table: The negative sampling table
        :type data: torch.Tensor
        :param batch_size: The current batch size
        :type batch_size: int
        :param targets: The target ids
        :type targets: torch.Tensor
        :return: The negative ids
        :rtype: torch.Tensor
        """
        negative_target_pos = torch.randint(
            len(expanded_ns_table),
            (batch_size, self.negative_samples),
        )

        # The ids of [self.negative_samples] negative targes (e.g. 2),
        # meaning targets that did not happen within the given context
        negative_ids = expanded_ns_table.take(negative_target_pos)

        # Check the negative ids do not overlap with the targets
        for i in range(len(targets)):
            while torch.any(torch.eq(negative_ids[i], targets[i])):
                # Resample for the row until there's no overlap
                new_samples = torch.randint(
                    len(expanded_ns_table),
                    (1, self.negative_samples),
                )
                negative_ids[i] = expanded_ns_table.take(new_samples)
        return negative_ids

    def save(self, filepath: str, filename: str) -> None:
        """Save the model to a file.

        In addition to the model, the model parameters, the train and validation loss,
        and the token to id mapping are saved in the designated filepath and with the
        designated filename.

        :param path: The path to the file.
        :type path: str
        :param filename: The filename.
        :type filename: str
        """
        assert self.train_epoch_loss is not [], "Train the model first."

        # Save the model parameters.
        hyperparams = {
            "embedding_size": self.embedding_size,
            "time_interval": self.time_interval,
            "time_scope_radius": self.time_scope_radius,
            "negative_samples": self.negative_samples,
            "loss_function": self.loss_function,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "device": self.device,
            "min_total_events": self.min_total_events,
            "max_events": self.max_events,
            "seed": self.seed,
        }

        # Save the model state_dict.
        embedding_state_dict = self.model.state_dict()

        # Save the train and validation loss.
        losses = {
            "train_loss": self.train_epoch_loss,
            "val_loss": self.validation_epoch_loss,
        }

        # Save the Token 2 ID dictionary.
        concept_name_to_idx_dict = self.data.concept_name_to_idx_dict

        # Save everything to a file.
        path = os.path.join(filepath, filename)

        # # Overwriting the file if it already exists.
        # if os.path.exists(path):
        #     os.remove(path)
        torch.save(
            {
                "hyperparams": hyperparams,
                "embedding_state_dict": embedding_state_dict,
                "losses": losses,
                "concept_name_to_idx_dict": concept_name_to_idx_dict,
            },
            path,
        )


class MCEwTAAModel(torch.nn.Module):
    """The MCEwTAA model. This model is a modification of the MCE model. It uses
    the time-aware attention mechanism from the TAA model.

    :param data: The data
    :type data: pd.DataFrame
    :param embedding_size: The embedding size
    :type embedding_size: int
    :param time_scope_radius: The time scope radius
    :type time_scope_radius: int
    :param device: The device
    :type device: torch.device
    :param crit: The loss criterion. Defaults to "bce_sigmoid". Can either take
        "bce_sigmoid" or "bce_softmax".
    :type crit: str
    """

    def __init__(
        self,
        data: DataClassMCE,
        embedding_size: int,
        time_scope_radius: int,
        device: torch.device,
        loss_function: str = "bce_sigmoid",
    ):
        super().__init__()
        self.data = data
        self.embedding_size = embedding_size
        self.context_embedding = torch.nn.Embedding(
            len(data.negative_sampling_table), embedding_size
        )
        self.target_embedding = torch.nn.Embedding(
            len(data.negative_sampling_table), embedding_size
        )
        self.softmax = torch.nn.Softmax(dim=1)
        self.taa_matrix = torch.nn.Parameter(
            torch.rand(len(data.negative_sampling_table), 2 * time_scope_radius + 1)
        )
        self.taa_bias = torch.nn.Parameter(
            torch.rand(len(data.negative_sampling_table), 1)
        )
        self.time_scope_radius = torch.tensor(time_scope_radius).to(device)
        self.device = device
        self.loss_function = loss_function

    def forward(
        self, context: torch.Tensor, mixed_target: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the model.

        :param context: The context ids and times
        :type context: torch.Tensor
        :param mixed_target: The target ids and times
        :type mixed_target: torch.Tensor
        :return: The scores
        :rtype: torch.Tensor
        """
        # --Steps--
        # 1. Prework
        # 2. Get embeddings per context id (give invalid ids (-1) any valid id (0)
        # 3. Calculate time aware factor per embedding
        # 4. Sum up all embeddings
        # 5. Get embeddings per target id
        # 6. Perform a Batch-Matrix-Multiplication to obtain the scores

        # Step 1:Prework
        # Split Context Event and Context Time
        # For the Neural Network, this is the input to determine which of
        # the targets it does belong to
        # Context is defined as the IDs of the events.
        context_event, context_time = (
            context[:, :, :1].squeeze(),
            context[:, :, 1:].squeeze(),
        )

        # n_options is the sum of positive target ids (1) and
        # negative (fake) target ids (negative_samples)
        n_options = mixed_target.shape[-1] - 1

        # Tensors are explicitly casted as LongTensor here in order to avoid weird
        # issues when using mps. And it needs then to be moved back to mps, as type
        # casting changes location
        target_event, target_time = (
            mixed_target[:, :n_options].type(torch.LongTensor).to(self.device),
            mixed_target[:, n_options:].type(torch.LongTensor).to(self.device),
        )

        # Step 2: Get embeddings per context id (give invalid ids (-1) any valid id (0)
        # some targets do not have "enough" context events surrounding it.
        # But as the tensor must be consistent in its shape,
        # missing context events are dentoted as -1
        # with context masks, we find these missing context events
        # and assign a mask value of 0 to them.
        # Thereby, they will not be relvant for the rest of the calculation
        context_mask = torch.where(context_event.detach().clone() == -1, 0, 1)
        context_event = context_event * context_mask

        # Get embedding for context
        context_embedding = self.context_embedding(context_event)

        # Step 3: Calculate time aware factor per embedding
        # this is clipped in order to not fall for invalid times (e.g. -1) if a valid
        # time in this badge is already quite high (e.g. 200)
        time_delta = torch.clip(
            context_time - target_time + self.time_scope_radius,
            0,
            self.time_scope_radius * 2,
        )

        # Calculate the time aware attention
        activated_taa_matrix = self.softmax(self.taa_matrix + self.taa_bias)

        taa_weights = activated_taa_matrix[context_event.squeeze(), time_delta]

        # Multiply each vector with the time aware attention
        # additionally, zero-out every irrelevant (-1) vector using the context_weights
        # use torch.unsqueeze instead of taa_weights[:, :, :, None] &
        # context_weights[:, :, :, None] as this is not yet supported when
        # using mps gpu device
        taa_weighted = (
            context_embedding * taa_weights.unsqueeze(-1) * context_mask.unsqueeze(-1)
        )

        # Step 4: Sum up all embeddings
        # sum up the context to retrieve the final hidden representation
        hidden_representation = torch.sum(taa_weighted, -2)

        # Step 5: Get embeddings per target id
        target_embedding = self.target_embedding(target_event)

        # Transpose the tensor for matrix multiplication
        # target_embedding = target_embedding.transpose(1, 2)

        # Step 6: Perform a Batch-Matrix-Multiplication to obtain the scores:
        # Multiply all of the targets (the one we want, and negative samples),
        # with the hidden representation of the context to get the scores.
        scores = torch.bmm(
            target_embedding, hidden_representation.unsqueeze(-1)
        ).squeeze()

        if self.loss_function == "bce_softmax":
            scores = self.softmax(scores.to(self.device))

        return scores

    def nearest_neighbors(
        self,
        concept_names: List[NodeIndex],
        n_neighbors: int = 3,
        translated: bool = False,
    ) -> List[NodeIndex]:
        """Find the nearest neighbors of a list of tokens.

        :param tokens: A list of tokens.
        :type tokens: list
        :param n_neighbors: The number of neighbors to find. Defaults to 3.
        :type n_neighbors: int, optional
        :param translated: Whether the tokens are already translated to id.
        :type translated: bool, optional
        :return: A list of lists of neighbors.
        :rtype: list
        """

        # Encode the tokens as integers, and put them into a PyTorch tensor.
        if translated:
            token_ix = torch.as_tensor(concept_names)
        else:
            token_ix = torch.as_tensor(
                [self.data.concept_name_to_idx(name) for name in concept_names]
            )

        # Look up the embeddings for the test tokens.
        voc_size, emb_dim = self.context_embedding.weight.shape
        test_emb = self.context_embedding(token_ix).view(len(concept_names), 1, emb_dim)

        # Also, get the embeddings for all tokens in the vocabulary.
        all_emb = self.context_embedding.weight.view(1, voc_size, emb_dim)

        # We'll use a cosine similarity function to find the most similar tokens.
        # The .view kludgery above is needed for the batch-wise cosine similarity.
        sim_func = torch.nn.CosineSimilarity(dim=2)
        scores = sim_func(test_emb, all_emb)
        # The shape of scores is (nbr of test tokens, total number of tokens)

        # Find the top-scoring columns in each row.
        near_nbr = scores.topk(n_neighbors + 1, dim=1)
        values = near_nbr.values[:, 1:]
        indices = near_nbr.indices[:, 1:]

        # Finally, map tokens indices back to strings, and put the result in a list.
        out = []
        for ixs, vals in zip(indices, values):
            out.append(
                [
                    (self.data.idx_to_concept_name(ix.item()), val.item())
                    for ix, val in zip(ixs, vals)
                ]
            )
        return out
