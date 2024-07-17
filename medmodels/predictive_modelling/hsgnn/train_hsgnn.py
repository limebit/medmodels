import logging
import time
from pathlib import Path
from typing import Tuple

import torch
import torch_geometric.transforms as T

from medmodels.dataclass.dataclass import MedRecord
from medmodels.predictive_modeling.hsgnn.hsgnn_model import (
    HSGNNModel,
    eval_link_predictor,
    train_link_predictor,
)
from medmodels.predictive_modeling.hsgnn.hsgnn_utils import (
    load_hyperparameters,
    load_medrecord,
    logger,
    save_model,
)
from medmodels.predictive_modeling.hsgnn.model_preprocessing import ModelPreprocessor

# Constants
REQUIRED_HYPERPARAMETERS = {
    "HIDDEN_CHANNELS",
    "OUT_CHANNELS",
    "NUM_EPOCHS",
    "LEARNING_RATE",
}


# TODO: change saving paths
# TODO: check if embeddings with mce change medrecord
def train_hsgnn(
    medrecord: MedRecord,
    data_path: Path,
    hyperparams_path: Path,
    embeddings_path: Path = None,
    mce_path: Path = None,
) -> Tuple:
    """
    Train HSGNN. Includes all steps from loading the graph, HSGNN preprocessing,
    to training the predictive model.

    :param data_path: the path to data
    :type data_path: string
    :param embeddings: path to embedding file
    :type embeddings: string, optional
    :param workers: the number of parallel processes to use for computation,
                    defaults to 1
    :type workers: int, optional
    :param mce: path to MCEwTAA model
    :type mce: string, optional
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    medrecord = load_medrecord(data_path)
    hyperparams = load_hyperparameters(
        hyperparams_path, REQUIRED_HYPERPARAMETERS, "model"
    )
    hidden_channels = hyperparams["HIDDEN_CHANNELS"]
    out_channels = hyperparams["OUT_CHANNELS"]
    num_epochs = hyperparams["NUM_EPOCHS"]
    learning_rate = hyperparams["LEARNING_RATE"]

    model_preprocessor = ModelPreprocessor(
        medrecord=medrecord,
        data_path=data_path,
        hyperparams_path=hyperparams_path,
        embeddings_path=embeddings_path,
        mce_path=mce_path,
    )
    data, subgraphs, node_types_tensor = model_preprocessor.prepare_data_for_model()

    logger.info("Splitting the dataset for training, validation and testing")
    start_time = time.time()
    split = T.RandomLinkSplit(
        num_val=0.05,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0,
    )
    train_data, val_data, test_data = split(data)
    end_time = time.time()
    logger.info(f"Splitted data in {end_time - start_time} seconds")

    model = HSGNNModel(
        in_channels=data.num_features,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        number_of_subgraphs=len(subgraphs),
        nodes_types=node_types_tensor,
    ).to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    logger.info("Training the model")
    start_time = time.time()
    model = train_link_predictor(
        model,
        train_data,
        val_data,
        optimizer,
        scheduler,
        criterion,
        subgraphs,
        num_epochs,
    )
    end_time = time.time()
    logger.info(f"Training time: {end_time - start_time} seconds")

    logger.info("Testing the model")
    start_time = time.time()
    node_embeddings, fused_matrix = model(test_data.x, test_data.edge_index, subgraphs)
    test_auc = eval_link_predictor(model, test_data, subgraphs)
    logger.info(f"Test: {test_auc:.3f}")
    final_edge_index = model.decode_all(node_embeddings, fused_matrix)
    end_time = time.time()
    logging.info(f"Testting time: {end_time - start_time} seconds")

    save_model(model, data_path, mce_path)
    return final_edge_index
