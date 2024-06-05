# Main branching point for evaluating on different datasets

from .fairness_eval import evaluate_dollar_street_dataset, evaluate_fairface_dataset, evaluate_geode_dataset
from .retr_eval import evaluate_retrieval_dataset
from .wds_eval import evaluate_webdataset
from .wilds_eval import evaluate_wilds_dataset
from .wino_eval import evaluate_winogavil_dataset


def evaluate_model(task_key, train_info, data_root, dataset_size, batch_size=64, prefix=None):
    if task_key.startswith("retrieval/"):
        metrics = evaluate_retrieval_dataset(
            task_key,
            train_info["scale_config"]["model"],
            train_info["checkpoint"],
            data_root=data_root,
            batch_size=batch_size,
            prefix=prefix,
        )
    elif task_key.startswith("wilds/"):
        metrics = evaluate_wilds_dataset(
            task_key,
            train_info["scale_config"]["model"],
            train_info["checkpoint"],
            data_root=data_root,
            dataset_len=dataset_size,
            batch_size=batch_size,
            prefix=prefix,
        )
    elif task_key.startswith("fairness/"):
        eval_fn = {
            "fairness/dollar_street": evaluate_dollar_street_dataset,
            "fairness/geode": evaluate_geode_dataset,
            "fairness/fairface": evaluate_fairface_dataset,
            "fairness/utkface": evaluate_fairface_dataset,
        }.get(task_key)
        if eval_fn is not None:
            metrics = eval_fn(
                task_key,
                train_info["scale_config"]["model"],
                train_info["checkpoint"],
                data_root=data_root,
                dataset_len=dataset_size,
                batch_size=batch_size,
                prefix=prefix,
            )
        else:
            metrics = {}
    elif task_key.startswith("misc/"):
        if task_key == "misc/winogavil":
            metrics = evaluate_winogavil_dataset(
                train_info["scale_config"]["model"],
                train_info["checkpoint"],
                data_root=data_root,
                batch_size=batch_size,
                prefix=prefix,
            )
        else:
            metrics = {}
    else:
        metrics = evaluate_webdataset(
            task_key,
            train_info["scale_config"]["model"],
            train_info["checkpoint"],
            data_root=data_root,
            dataset_len=dataset_size,
            batch_size=batch_size,
            prefix=prefix,
        )
    return metrics
