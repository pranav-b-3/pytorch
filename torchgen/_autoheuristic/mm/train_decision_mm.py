# mypy: ignore-errors
import sys
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from typing import Any, Callable
Value = Any


sys.path.append(str(Path(__file__).absolute().parents[1]))

# from train_regression_mem import AHTrainRegressionTree
from train_decision_mem import AHTrainDecisionTree

# from torch._inductor.autoheuristic.autoheuristic_utils import mm_operations

class AHOperation:
    """
    AHOperation can be used to augment the data collected by AutoHeuristic.
    One might for example store features like m, k, n, but also want to use
    features like m*n, or k*n, to learn a heuristic. Instead of storing features
    that can be created from the collected data, one can use AHOperation to
    create new features from the collected data.
    """

    def __init__(
        self, name: str, func: Callable[[Any], Value], is_categorical: bool = False
    ) -> None:
        self.name = name
        self.func = func
        self.is_categorical = is_categorical

    def apply_operation(self, data: Any) -> None:
        data[self.name] = self.func(data)

def get_arith_intensity(data: Any) -> float:
    m = data["m"]
    k = data["k"]
    n = data["n"]
    if m == 0 or k == 0 or n == 0:
        return 0.0
    return m * k * n / (m * k + k * n + m * n)

def get_mult_dims_ops() -> list[AHOperation]:
    m_times_k_op = AHOperation("m*k", lambda data: data["m"] * data["k"])
    m_times_n_op = AHOperation("m*n", lambda data: data["m"] * data["n"])
    k_times_n_op = AHOperation("k*n", lambda data: data["k"] * data["n"])
    return [m_times_k_op, m_times_n_op, k_times_n_op]
        
def mm_operations() -> list[AHOperation]:
    mult_dims_ops = get_mult_dims_ops()
    arith_intensity_op = AHOperation("arith_intensity", get_arith_intensity)
    return mult_dims_ops + [arith_intensity_op]


class AHTrainDecisionTreeMM(AHTrainDecisionTree):
    def __init__(self):
        super().__init__()

    def add_new_features(self, results):
        ops = mm_operations()
        added_categorical_features = []
        for op in ops:
            results[op.name] = results.apply(op.func, axis=1)
            if op.is_categorical:
                added_categorical_features.append(op.name)
        return (results, added_categorical_features)

    def get_default_config(self, row):
        return "extern_mm"

    def get_allowed_wrong_prediction_pct(self):
        return 1.0

    def get_test_and_val_size(self):
        return (0.15, 0.15)

    # def get_grid_search_values(self):
    #     return {"max_depth": [5], "min_samples_leaf": [0.01], "criterion": ["entropy"]}

    def add_training_data(self, df_train, datasets):
        # add each dataset to the training data 3 times
        # we really want to make sure that the heuristic performs well on these datasets
        # df_timm_train = datasets["train_timm"]
        # df_timm_train = df_timm_train.loc[df_timm_train.index.repeat(3)].reset_index(
        #     drop=True
        # )
        # df_hf_train = datasets["train_hf"]
        # df_hf_train = df_hf_train.loc[df_hf_train.index.repeat(3)].reset_index(
        #     drop=True
        # )
        df_train = datasets["train"]
        df_train= df_train.loc[df_train.index.repeat(3)].reset_index(
            drop=True
        )
        # df_train = pd.concat(
        #     [df_train, df_timm_train, df_hf_train],
        #     ignore_index=True,
        # )
        return df_train

    def ranking_always_included_choices(self):
        return ["extern_mm"]


if __name__ == "__main__":
    train = AHTrainDecisionTreeMM()
    train.generate_heuristic()
