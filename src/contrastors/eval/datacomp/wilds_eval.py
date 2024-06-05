# For evaluation of WILDS datasets

import os

import numpy as np
import torch
import webdataset as wds
from tqdm import tqdm


# Replace wilds function that requires torch_scatter
def _avg_over_groups(v, g, n_groups):
    """
    Args:
        v (Tensor): Vector containing the quantity to average over.
        g (Tensor): Vector of the same length as v, containing group information.
    Returns:
        group_avgs (Tensor): Vector of length num_groups
        group_counts (Tensor)
    """
    assert v.device == g.device
    assert v.numel() == g.numel()
    group_count = wilds.common.utils.get_counts(g, n_groups)
    # group_avgs = torch_scatter.scatter(src=v, index=g, dim_size=n_groups, reduce='mean')
    group_avgs = torch.zeros(n_groups, dtype=torch.float, device=v.device).scatter_(0, index=g, src=v, reduce="add")
    group_avgs /= group_count
    return group_avgs, group_count


import wilds.common.utils

wilds.common.utils.avg_over_groups = _avg_over_groups
#

from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import F1, Accuracy, Recall
from wilds.datasets.wilds_dataset import WILDSDataset

from .wds_eval import create_webdataset, evaluate_webdataset


def create_metadata_loader(task, data_root=None, dataset_len=None, batch_size=64, num_workers=4):
    dataset, _ = create_webdataset(task, None, data_root, dataset_len, batch_size, num_workers)
    # Load metadata (npy) and no images
    dataset.pipeline = dataset.pipeline[:5]  # This will break if webdataset changes
    dataset = dataset.to_tuple("cls", "npy")
    if dataset_len:
        dataset = dataset.with_length((dataset_len + batch_size - 1) // batch_size)

    dataloader = torch.utils.data.DataLoader(
        dataset.batched(batch_size),
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
    )
    return dataloader


def evaluate_wilds_dataset(
    task,
    model_arch,
    model_path,
    data_root=None,
    dataset_len=None,
    batch_size=64,
    num_workers=4,
    prefix=None,
):
    """Evaluate CLIP model on WILDS classification task."""

    # Evaluate
    metrics, y_pred, y_target = evaluate_webdataset(
        task,
        model_arch,
        model_path,
        data_root,
        dataset_len,
        batch_size,
        num_workers,
        return_preds=True,
        prefix=prefix,
    )

    # Load additional metadata
    print("Reading additional metadata")
    metadata_loader = create_metadata_loader(task, data_root, dataset_len, batch_size, num_workers)
    # Check metadata
    y_array = []
    metadata_array = []
    for label, metadata in metadata_loader:
        y_array.append(label)
        metadata_array.append(metadata)
    # assert (y_target == np.array(y_array)).all(), "Labels do not match"
    metadata = torch.cat(metadata_array)

    # Compute additional metrics
    wilds_evaluator = EVALUATORS[task](metadata)
    metrics.update(wilds_evaluator.eval(y_pred, y_target, metadata)[0])

    return metrics


# WILDS


class WILDSEvaluator(WILDSDataset):
    def __init__(self, metadata):
        self._metadata_array = metadata


# iWildCam


class IWildCamEvaluator(WILDSEvaluator):
    def __init__(self, metadata):
        super().__init__(metadata)
        self._metadata_fields = [
            "location",
            "sequence",
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "y",
        ]
        self._eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=(["location"]))

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        metrics = [
            Accuracy(prediction_fn=prediction_fn),
            Recall(prediction_fn=prediction_fn, average="macro"),
            F1(prediction_fn=prediction_fn, average="macro"),
        ]
        results = {}
        for metric in metrics:
            results.update(
                {
                    **metric.compute(y_pred, y_true),
                }
            )
        results_str = (
            f"Average acc: {results[metrics[0].agg_metric_field]:.3f}\n"
            f"Recall macro: {results[metrics[1].agg_metric_field]:.3f}\n"
            f"F1 macro: {results[metrics[2].agg_metric_field]:.3f}\n"
        )
        return results, results_str


# Camelyon17


class Camelyon17Evaluator(WILDSEvaluator):
    def __init__(self, metadata):
        super().__init__(metadata)
        self._metadata_fields = ["hospital", "slide", "y"]
        self._eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=["slide"])

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        metric = Accuracy(prediction_fn=prediction_fn)
        return self.standard_group_eval(metric, self._eval_grouper, y_pred, y_true, metadata)


# FMoW


class FMoWEvaluator(WILDSEvaluator):
    def __init__(self, metadata):
        super().__init__(metadata)
        self._metadata_fields = ["region", "year", "y"]
        self._eval_groupers = {
            "year": CombinatorialGrouper(dataset=self, groupby_fields=["year"]),
            "region": CombinatorialGrouper(dataset=self, groupby_fields=["region"]),
        }

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        metric = Accuracy(prediction_fn=prediction_fn)
        # Overall evaluation + evaluate by year
        all_results, all_results_str = self.standard_group_eval(
            metric, self._eval_groupers["year"], y_pred, y_true, metadata
        )
        # Evaluate by region and ignore the "Other" region
        region_grouper = self._eval_groupers["region"]
        region_results = metric.compute_group_wise(
            y_pred,
            y_true,
            region_grouper.metadata_to_group(metadata),
            region_grouper.n_groups,
        )
        all_results[f"{metric.name}_worst_year"] = all_results.pop(metric.worst_group_metric_field)
        region_metric_list = []
        for group_idx in range(region_grouper.n_groups):
            group_str = region_grouper.group_field_str(group_idx)
            group_metric = region_results[metric.group_metric_field(group_idx)]
            group_counts = region_results[metric.group_count_field(group_idx)]
            all_results[f"{metric.name}_{group_str}"] = group_metric
            all_results[f"count_{group_str}"] = group_counts
            if region_results[metric.group_count_field(group_idx)] == 0 or "Other" in group_str:
                continue
            all_results_str += (
                f"  {region_grouper.group_str(group_idx)}  "
                f"[n = {region_results[metric.group_count_field(group_idx)]:6.0f}]:\t"
                f"{metric.name} = {region_results[metric.group_metric_field(group_idx)]:5.3f}\n"
            )
            region_metric_list.append(region_results[metric.group_metric_field(group_idx)])
        all_results[f"{metric.name}_worst_region"] = metric.worst(region_metric_list)
        all_results_str += f"Worst-group {metric.name}: {all_results[f'{metric.name}_worst_region']:.3f}\n"

        return all_results, all_results_str


EVALUATORS = {
    "wilds/iwildcam": IWildCamEvaluator,
    "wilds/camelyon17": Camelyon17Evaluator,
    "wilds/fmow": FMoWEvaluator,
}
