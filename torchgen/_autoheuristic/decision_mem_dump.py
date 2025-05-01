# mypy: ignore-errors

import itertools
import json
import logging
import math
import warnings


warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
)

from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from ah_tree import DecisionTree
from scipy.stats import gmean
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from train import AHTrain

ALPHA=2
BETA=1

log = logging.getLogger(__name__)
DEBUG = True
if DEBUG:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    log.addHandler(ch)


class AHTrainDecisionTree(AHTrain):
    def __init__(self):
        super().__init__()

    def debug_time(self, row, top_k_choices):
        choices_feedback = json.loads(row["choice2time"])
        timings = sorted(choices_feedback.items(), key=lambda x: x[1])
        for choice, time in timings:
            result = f"{choice} {time}"
            if choice in top_k_choices:
                result += " TOPK"
            print(result)

    def is_unsafe_leaf(self, row, predicted_config, choice2time):
        """
        Can be overridden by subclasses to define their own logic for deciding when a leaf is unsafe. Returns a sample
        that landed in the leaf, the choice predicted by the tree, and a dictionary that maps each choice to the
        execution time. One can for example decide to mark a leaf as unsafe if the predicted choice is 2x slower
        than the fastest choice.
        If a leaf is unsafe, the learned heuristic will always return 'unsure' if an input lands in that leaf.
        """

        return False

    def get_unsafe_leaves(self, model, df, feature_columns):
        """
        Given a trained decision tree, and a dataframe containing the training data, returns a list of unsafe leaves.
        """
        X = df[feature_columns]
        leaf_ids = model.apply(X)
        unique_leaves = np.unique(leaf_ids)

        unsafe_leaves = []
        # Iterate over each leaf
        for leaf in unique_leaves:
            leaf_mask = leaf_ids == leaf
            # Get samples that land in this leaf
            leaf_X = X[leaf_mask]

            predicted_config = model.predict(leaf_X.iloc[[0]])[0]

            # For each sample, check if we should mark the leaf as unsafe
            for idx, row in leaf_X.iterrows():
                choice2time = json.loads(df.loc[idx, "choice2time"])
                if self.is_unsafe_leaf(row, predicted_config, choice2time):
                    unsafe_leaves.append(leaf)
                    break
        return unsafe_leaves

    def get_allowed_wrong_prediction_pct(self):
        """
        This is used to determine a threshold for when a learned heuristic returns 'unsure'.
        If this function returns 0.01, we will set the probability required for the decision tree to return a decision
        such that at most 1% of the predictions will be wrong on the validation set.
        """
        return 0.01

    def get_grid_search_values(self):
        """
        Standard values for grid search. Can be overriden.
        """
        return {
            "max_depth": [5, 6, 7],
            "min_samples_leaf": [1, 5, 10, 0.01, 0.05, 0.02],
            "criterion": ["gini", "entropy"],
        }

    def predict(self, model, df, feature_columns):
        """
        Returns the predictions, probabilities, and leaf ids for a given dataframe.
        """
        predictions = model.predict(df[feature_columns])
        proba = model.predict_proba(df[feature_columns])
        leaf_ids = model.apply(df[feature_columns])
        return predictions, proba, leaf_ids

    def train_and_evaluate_models(
        self,
        datasets,
        max_depths,
        min_samples_leafs,
        criterion_list,
        feature_columns,
    ):
        """
        Does a grid search over max_depths, min_samples_leafs, and criterion_list and returns the best model.
        """

        results = []
        best_model = None
        best_model_safe_proba = 0
        best_model_num_correct = 0
        best_model_unsafe_leaves = []
        columns = ["set", "crit", "max_depth", "min_samples_leaf"]
        metrics_columns = []
        for max_depth, min_samples_leaf, criterion in itertools.product(
            max_depths, min_samples_leafs, criterion_list
        ):
            print(
                f"max_depth={max_depth} min_samples_leaf={min_samples_leaf} criterion={criterion}"
            )
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                random_state=42,
            )
            df_train = datasets["train"]
            df_val = datasets["val"]
            model.fit(df_train[feature_columns], df_train["winner"])

            model = DecisionTree(model, feature_columns)

            unsafe_leaves = self.get_unsafe_leaves(model, df_train, feature_columns)
            predictions, proba, leaf_ids = self.predict(model, df_val, feature_columns)

            wrong_pct = self.get_allowed_wrong_prediction_pct()
            evaluator = DecisionEvaluator(
                self,
                model,
                predictions,
                df_val,
                proba,
                wrong_pct=wrong_pct,
                unsafe_leaves=unsafe_leaves,
                leaf_ids=leaf_ids,
            )
            safe_proba = evaluator.get_safe_proba()
            print(f"safe_proba={safe_proba}")

            def eval(name, df):
                predictions, proba, leaf_ids = self.predict(model, df, feature_columns)
                evaluator = DecisionEvaluator(
                    self,
                    model,
                    predictions,
                    df,
                    proba,
                    wrong_pct=wrong_pct,
                    threshold=safe_proba,
                    unsafe_leaves=unsafe_leaves,
                    leaf_ids=leaf_ids,
                )
                return evaluator.get_results()

            for dataset_name, dataset in datasets.items():
                eval_result: EvalResults = eval(dataset_name, dataset)
                eval_result_metrics = eval_result.to_map()
                if dataset_name == "val":
                    num_correct = eval_result.accuracy.num_correct
                    num_wrong = eval_result.accuracy.num_wrong
                    num_total = eval_result.accuracy.total
                    if num_wrong <= num_total * wrong_pct:
                        if num_correct > best_model_num_correct:
                            print(
                                f"new best model with {num_correct} correct and {num_wrong} wrong"
                            )
                            best_model = model
                            best_model_num_correct = num_correct
                            best_model_safe_proba = safe_proba
                            best_model_unsafe_leaves = unsafe_leaves

                result = (dataset_name, criterion, max_depth, min_samples_leaf)
                result += tuple(eval_result_metrics.values())
                results.append(result)
                if len(metrics_columns) == 0:
                    metrics_columns = list(eval_result_metrics.keys())
                    columns += metrics_columns

        return (
            pd.DataFrame(results, columns=columns),
            best_model,
            best_model_safe_proba,
            best_model_unsafe_leaves,
        )

    def get_test_and_val_size(self):
        """
        Returns the size of the test and validation sets.
        """
        return (0.15, 0.15)

    def prepare_datasets(self, df, other_datasets, cat_feature2cats):
        """
        Splits the dataframe into train, val, and test sets.
        Also adds other datasets, specified by the user, to the train set.
        """
        test_size, val_size = self.get_test_and_val_size()
        # Split into train+val and test
        df_train_val, df_test = train_test_split(
            df, test_size=test_size, random_state=42
        )

        # Split train+val inputs into train and val
        train_val_size = 1 - test_size
        df_train, df_val = train_test_split(
            df_train_val, test_size=val_size / train_val_size, random_state=42
        )
        datasets = {"train": df_train, "val": df_val, "test": df_test}
        self.add_real_datasets(datasets, other_datasets, cat_feature2cats)
        return datasets

    def export_to_dot(self, best_model, df, feature_columns):
        """
        Export a learned decision tree to a dot file.
        """
        dot_str = best_model.to_dot()
        with open("best_model.dot", "w") as f:
            f.write(dot_str)

    def get_feature_columns(self, df):
        """
        The dataframe contains columns that are not features, such as 'winner', 'speedup' that are only used for
        debugging purposes. This function returns the columns that are actually features.
        """
        exclude_columns = [
            "speedup",
            "winner",
            "target",
            "avail_choices",
            "choice2time",
            "index",
            "actual_winner",
            "relative_performance",
        ]
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        return feature_columns

    def add_training_data(self, df_train, datasets):
        return datasets["train"]

    def main(
        self,
        log_path,
        other_datasets,
        nrows,
        heuristic_name,
        save_dot=False,
    ):
        """
        Main function that trains a decision tree and generates a heuristic.
        """
        # TODO: Enable apply_filters
        (df, choices, cat_feature2cats, dummy_col_2_col_val, metadata) = self.get_df(
            log_path, nrows=nrows, apply_filters=False,
        )
        self.dummy_col_2_col_val = dummy_col_2_col_val
        datasets = self.prepare_datasets(df, other_datasets, cat_feature2cats)
        df_train = self.add_training_data(datasets["train"], datasets)
        datasets["train"] = df_train
        # print(datasets["train"]["winner"].value_counts().to_string())

        feature_columns = self.get_feature_columns(df)
        grid_search_values = self.get_grid_search_values()
        max_depths = grid_search_values["max_depth"]
        min_samples_leafs = grid_search_values["min_samples_leaf"]
        criterion_list = grid_search_values["criterion"]
        (
            results_df,
            best_model,
            best_model_safe_proba,
            unsafe_leaves,
        ) = self.train_and_evaluate_models(
            datasets,
            max_depths,
            min_samples_leafs,
            criterion_list,
            feature_columns,
        )

        # prints results for all models and datasets
        print(results_df.to_string())

        sort_metric = "correct"
        # prints results grouped by dataset
        # for set_name in results_df["set"].unique():
        #     dataset_results = results_df[results_df["set"] == set_name]
        #     dataset_results = dataset_results.sort_values(by=sort_metric)
        #     print(dataset_results.to_string() + "\n")

        if best_model is not None:
            if save_dot:
                self.export_to_dot(best_model, df, feature_columns)
            self.codegen(
                best_model,
                metadata,
                heuristic_name,
                best_model_safe_proba,
                dummy_col_2_col_val,
                unsafe_leaves,
            )
        else:
            print(
                "All learned models have too many wrong predictions, so no heuristic was generated"
            )

    def get_df(
        self,
        log_path,
        cat_feature2cats=None,
        nrows=None,
        apply_filters=False,
    ):
        """
        Parses the log file and processes the data into a dataframe that can be used for training.
        """
        (df, metadata, features, categorical_features, choices) = self.parse_log(
            log_path, nrows
        )
        
        # df = df[df["sm_utilization_after"] > 0]
        # print(df.groupby('choice')['sm_utilization_after'].std().mean())
        
        df["score"] = ALPHA * df["feedback"] + BETA * df["memory_utilization_after"]
        
        def calculate_stats(group):
            count = len(group)

            # for stat_col in ["feedback", "sm_utilization_after" "memory_utilization_before,memory_u"]
            has_inf = np.isinf(group["feedback"]).any()
            if has_inf:
                relative_std = np.inf
                median = np.inf
            else:
                mean = group["feedback"].mean()
                std = group["feedback"].std()
                relative_std = (std / mean) * 100 if mean != 0 else np.inf
                median = group["feedback"].median()
            if relative_std > 5:
                times = group["feedback"].tolist()
                times_str = ", ".join([f"{t:.3f}" for t in sorted(times)])
                log.debug("High relative std: %f. times=%s", relative_std, times_str)                
            
            has_inf = np.isinf(group["sm_utilization_after"]).any()
            if has_inf:
                relative_std_sm = np.inf
                median_sm = np.inf
            else:
                mean_sm = group["sm_utilization_after"].mean()
                std_sm = group["sm_utilization_after"].std()
                relative_std_sm = (std / mean) * 100 if mean != 0 else np.inf
                median_sm = group["sm_utilization_after"].median()
            if relative_std > 5:
                times_sm = group["sm_utilization_after"].tolist()
                
                
            has_inf = np.isinf(group["memory_utilization_after"]).any()
            if has_inf:
                relative_std_mem = np.inf
                median_mem = np.inf
            else:
                mean_mem = group["memory_utilization_after"].mean()
                std_mem = group["memory_utilization_after"].std()
                relative_std_mem = (std / mean) * 100 if mean != 0 else np.inf
                median_mem = group["memory_utilization_after"].median()
            if relative_std > 5:
                times_mem = group["memory_utilization_after"].tolist()
                
            has_inf = np.isinf(group["score"]).any()
            if has_inf:
                relative_std_scpre = np.inf
                median_score = np.inf
            else:
                mean_score = group["score"].mean()
                std_score = group["score"].std()
                relative_std_score = (std / mean) * 100 if mean != 0 else np.inf
                median_score = group["score"].median()
            if relative_std > 5:
                times_score = group["score"].tolist()
                                                
            return pd.Series(
                {
                    "count": count,
                    "relative_std": relative_std,
                    "median_execution_time": median,
                    "relative_std_sm": relative_std_sm,
                    "median_sm": median_sm,
                    "relative_std_mem": relative_std_mem,
                    "median_mem": median_mem,
                    "median_sm_delta": group["sm_utilization_after"].median() - group["sm_utilization_before"].median()
                    }
            )

        feature_columns = features
        stats = (
            df.groupby(feature_columns + ["choice"], as_index=False)
            .apply(calculate_stats, include_groups=False)
            .reset_index()
        )

        # TODO: We have to be careful with removing certain choices, because if we e.g. remove the winner, the
        # heuristic will end up learning wrong things. But, execution times with high variance are also bad
        if apply_filters:
            # Filter out inputs with less than 3 measurements or high relative std
            valid_stats = stats[(stats["count"] >= 3) & (stats["relative_std"] <= 5)]
            # Group by input features and count how many valid choices we have for each input
            valid_inputs = valid_stats.groupby(feature_columns).filter(
                lambda x: len(x) >= 2
            )
        else:
            valid_inputs = stats

        # Compute the winner and speedup for each valid input
        
        def get_winner_and_speedup(group):
            assert len(group) >= 2, "Need at least 2 choices"
            
            # OBJECTIVE = "median_execution_time"
            # OBJECTIVE = "median_sm"
            OBJECTIVE = "median_mem"
            # OBJECTIVE = "median_sm_delta"


            sorted_group = group.sort_values(OBJECTIVE, ascending=True)
            winner = sorted_group.iloc[0]["choice"]
            winning_time = sorted_group.iloc[0][OBJECTIVE]
            second_best_time = sorted_group.iloc[1][OBJECTIVE]
            unique_choices = group["choice"].unique()

            choice2time = {}
            obj_index = list(group.columns).index(OBJECTIVE)
            for row in group.itertuples():
                choice2time[row.choice] = row[obj_index]
                
            
            assert len(unique_choices) == len(group), (
                f"len(unique_choices) != len(group): {len(unique_choices)} != {len(group)}"
            )

            return pd.Series(
                {
                    "winner": winner,
                    "avail_choices": unique_choices,
                    "choice2time": json.dumps(choice2time),
                }
            )

        results = (
            valid_inputs.groupby(feature_columns, as_index=False)
            .filter(lambda x: len(x) >= 2)
            .groupby(feature_columns, as_index=False)
            .apply(get_winner_and_speedup, include_groups=False)
            .reset_index()
        )

        (results, added_categorical_features) = self.add_new_features(results)
        categorical_features += added_categorical_features

        (
            results,
            cat_feature2cats,
            dummy_col_2_col_val,
        ) = self.handle_categorical_features(
            cat_feature2cats, categorical_features, results
        )
        return (results, choices, cat_feature2cats, dummy_col_2_col_val, metadata)

    def gen_classes(self, classes, num_spaces):
        """
        If classes=['choice1', 'choice2', 'choice3'], then this function returns
        the following string:
        self.choices.append('choice1')
        self.choices.append('choice2')
        self.choices.append('choice3')
        Used in the generated heuristic to map the index of a choice to its name.
        """
        indent = " " * num_spaces
        return "\n".join([f"{indent}self.choices.append('{c}')" for c in classes])

    def get_default_config(self, row):
        """
        Returns the default config for a given sample. The default config could for example be the config that is
        the chosen by a current handwritten heuristic. This can for example be used in get_unsafe_leaf to
        compare the predicted config with the default config.
        """
        return None

    def gen_predict_fn_def(self):
        """
        Generates the definition of the predict function.
        """
        return "def get_best_choices(self, context: AHContext) -> Optional[list[tuple[float, int]]]:"

    def codegen_boilerplate(
        self, heuristic_name, opt_name, threshold, shared_memory, device_capa, classes
    ):
        """
        Generates the boilerplate code for the generated heuristic. This includes things like imports, class definition,
        etc.
        """

        boiler_plate = f"""# flake8: noqa: B950
# fmt: off
# This file was generated by AutoHeuristic. Do not modify it manually!
# To regenerate this file, take a look at the steps in the README.md file inside torchgen/_autoheuristic/{opt_name}/
from typing import Optional

from torch._inductor.autoheuristic.autoheuristic_utils import (
    AHContext,
    AHMetadata,
    Choice,
)
from torch._inductor.autoheuristic.learnedheuristic_interface import (
    LearnedHeuristicDecision,
)


class {heuristic_name}(LearnedHeuristicDecision):

    def __init__(self) -> None:
        self.choices: list[Choice] = []
        self.fill_choices()

{self.gen_precondition(opt_name, shared_memory, device_capa)}

    def get_confidence_threshold(self) -> float:
        return {threshold}

    def get_choice(self, idx: int) -> Optional[str]:
        if idx < len(self.choices):
            return self.choices[idx]
        return None

    def fill_choices(self) -> None:
{self.gen_classes(classes, num_spaces=8)}

    def get_name(self) -> str:
        return '{opt_name}'"""
        return boiler_plate

    def add_real_datasets(
        self, datasets, other_datasets, cat_feature2cats
    ):
        """
        Adds datasets specified by the user to the datasets dictionary.
        """
        if other_datasets:
            for name, path in other_datasets:
                (df_other, choices, _, _, _) = self.get_df(
                    path,
                    cat_feature2cats=cat_feature2cats,
                    apply_filters=False,
                )
                datasets[name] = df_other

    def codegen(
        self,
        tree,
        metadata,
        heuristic_name,
        threshold,
        dummy_col_2_col_val,
        unsafe_leaves,
    ):
        lines = []
        device_capa = metadata["device_capa"]
        device_capa_str = f"({device_capa[0]}, {device_capa[1]})"
        opt_name = metadata["name"]
        lines.append(
            self.codegen_boilerplate(
                heuristic_name,
                opt_name,
                threshold,
                metadata["shared_memory"],
                device_capa_str,
                tree.classes_,
            )
        )
        fn_def = f"\n    {self.gen_predict_fn_def()}"
        lines.append(fn_def)
        tree.codegen(dummy_col_2_col_val, lines, unsafe_leaves)
        self.write_heuristic_to_file(lines, heuristic_name)


@dataclass
class AccuracyMetrics:
    # Number of correct predictions
    num_correct: int
    # Number of wrong predictions
    num_wrong: int
    # Number of predictions where model is unsure
    num_unsure: int
    # Total number of predictions
    total: int

    def to_map(self):
        return {
            "correct": self.num_correct,
            "wrong": self.num_wrong,
            "unsure": self.num_unsure,
            "total": self.total,
        }

@dataclass
class EvalResults:
    accuracy: AccuracyMetrics

    def to_map(self):
        return {
            **self.accuracy.to_map(),
        }


class DecisionEvaluator:
    def __init__(
        self,
        train,
        model,
        predictions,
        df,
        probas,
        wrong_pct=0.01,
        threshold=0.0,
        k=10,
        unsafe_leaves=None,
        leaf_ids=None,
    ) -> None:
        self.train = train
        self.model = model
        self.predictions = predictions
        self.df = df
        self.probas = probas
        self.wrong_pct = wrong_pct
        self.threshold = threshold
        self.k = k
        self.unsafe_leaves = unsafe_leaves
        self.leaf_ids = leaf_ids

        self.num_correct = 0
        self.num_wrong = 0
        self.num_unsure = 0
        self.wrong_probas = []
        self.speedups_wrong = []
        self.num_correct_top_k = 0
        self.num_wrong_top_k = 0
        self.wrong_speedups_top_k = []
        self.top_k_unsure = 0
        self.num_non_default_predictions = 0
        self.speedups_over_default = []
        self.num_default_better = 0

    def get_time(self, row, choice):
        choices_feedback = json.loads(row["choice2time"])
        return choices_feedback.get(choice, None)

    def eval_prediction(
        self, avail_choices, leaf_id, pred, true, prob, threshold, default_config, i
    ):
        max_prob = max(prob)
        if (
            leaf_id in self.unsafe_leaves
            or pred not in avail_choices
            or (max_prob != 1.0 and max_prob <= threshold)
        ):
            self.num_unsure += 1
        elif pred == true:
            self.num_correct += 1
        else:
            self.num_wrong += 1
            self.wrong_probas.append(max_prob)

    def get_safe_proba(self):
        return self.get_results(return_safe_proba=True)

    def compute_safe_proba(self, num_predictions, wrong_probas, wrong_pct):
        wrong_probas.sort()
        num_wrong = len(wrong_probas)
        allowed_wrong = int(num_predictions * wrong_pct)
        if allowed_wrong >= num_wrong:
            return 0.0
        too_many_wrong = num_wrong - allowed_wrong
        idx = min(too_many_wrong, len(wrong_probas) - 1)
        return wrong_probas[idx]

    def get_results(self, return_safe_proba=False) -> EvalResults:
        """
        Custom evaluation function that evaluates a learned decision tree.
        """

        y_true = self.df["winner"]
        for i, (pred, true, prob, leaf_id) in enumerate(
            zip(self.predictions, y_true, self.probas, self.leaf_ids)
        ):
            avail_choices = self.df["avail_choices"].iloc[i]
            assert true in avail_choices, (
                f"Best choice {true} not in available choices {avail_choices}"
            )
            default_config = self.train.get_default_config(self.df.iloc[i])
            self.eval_prediction(
                avail_choices,
                leaf_id,
                pred,
                true,
                prob,
                self.threshold,
                default_config,
                i,
            )

        total = len(self.predictions)
        if return_safe_proba:
            return self.compute_safe_proba(total, self.wrong_probas, self.wrong_pct)

        accuracyMetrics = AccuracyMetrics(
            self.num_correct, self.num_wrong, self.num_unsure, total
        )
        return EvalResults(
            accuracyMetrics,
        )


if __name__ == "__main__":
    train = AHTrainDecisionTree()
    train.generate_heuristic()