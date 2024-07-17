import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional, Literal

import torch

from medmodels import MedRecord


def setup_logger() -> logging.Logger:
    """
    Sets up the logger.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return logger


logger = setup_logger()


def load_hyperparameters(
    hyperparams_path: Path,
    hyperparams_type: Literal["model", "preprocessing"],
) -> Dict[str, Any]:
    """
    Loads hyperparameters from config file.

    Args:
        hyperparams_path (Path): Path to the hyperparameters file.
        hyperparams_type (Literal["model", "preprocessing"]): Type of hyperparameters to load.

    Returns:
        Dict[str, Any]: Loaded hyperparameters.

    Raises:
        FileNotFoundError: If the hyperparameters file does not exist.
        KeyError: If the hyperparameters type is not found in the file.
    """
    if not hyperparams_path.exists():
        msg = f"Could not find hyperparameters file at {hyperparams_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    with open(hyperparams_path, "r") as f:
        hyperparameters = json.load(f)
    try:
        hyperparameters[hyperparams_type]
    except KeyError:
        msg = f"{hyperparams_type} hyperparameters not found in {hyperparams_path}"
        logger.error(msg)
        raise KeyError(msg)

    logger.info(f"Hyperparameters loaded from '{hyperparams_path}'")
    return hyperparameters[hyperparams_type]


def load_medrecord(data_path: Path) -> MedRecord:
    """
    Loads medrecord from the given path.

    Args:
        data_path (Path): The path to the medrecord data.

    Returns:
        MedRecord: The loaded medrecord.

    Raises:
        FileNotFoundError: If the data file does not exist.
    """
    logger.info("Loading graph")
    start_time = time.time()
    if not data_path.exists():
        msg = f"File {data_path} does not exist"
        logger.error(msg)
        raise FileNotFoundError(msg)

    with open(data_path, "rb") as f:
        medrecord = pickle.load(f)

    end_time = time.time()
    logger.info(f"Loaded graph in {end_time - start_time} seconds")
    return medrecord


def save_model(model, data_path: Path, mce_path: Optional[Path] = None) -> None:
    """
    Saves the model to the given path.
    :param model: The model to save.
    :type model: torch.nn.Module
    :param model_path: The path to original data.
    :type model_path: Path
    """
    file_name = os.path.basename(data_path)
    dir_name = os.path.dirname(data_path) + "/model/"
    os.makedirs(os.path.dirname(dir_name), exist_ok=True)
    if not mce_path:
        model_path = dir_name + "model_" + file_name
    else:
        model_path = dir_name + "model_mce_" + file_name
    logger.info(f"Saving model to {model_path}")
    try:
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to '{model_path}'")
    except Exception as e:
        msg = f"Could not save model to '{model_path}': {e}"
        logger.error(msg)
        raise
