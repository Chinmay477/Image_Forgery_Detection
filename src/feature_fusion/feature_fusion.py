import math

import torch
import numpy as np


def get_yi(model, patch):

    with torch.no_grad():
        model.eval()
        return model(patch)


class WrongOperationOption(Exception):
    pass


def get_y_hat(y: np.ndarray, operation: str):

    if operation == "max":
        return np.array(y).max(axis=0, initial=-math.inf)
    elif operation == "mean":
        return np.array(y).mean(axis=0)
    else:
        raise WrongOperationOption("The operation can be either mean or max")
