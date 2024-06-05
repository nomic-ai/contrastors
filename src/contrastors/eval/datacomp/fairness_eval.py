from collections import defaultdict
from contextlib import suppress

from .wds_eval import *
from .wilds_eval import *

# Dollar Street


class TopKAccuracy(Accuracy):
    def __init__(self, prediction_fn=None, name=None):
        if name is None:
            name = "acc_topk"
        super().__init__(name=name)

    def _compute_element_wise(self, y_pred, y_true):
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)
        return (y_pred == y_true.unsqueeze(-1)).any(-1).float()


class DollarStreetEvaluator(WILDSEvaluator):
    def __init__(self, metadata):
        super().__init__(metadata)
        self._metadata_fields = ["income_ds", "income_meta", "region"]
        self._eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=["income_ds"])

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        metric = TopKAccuracy(prediction_fn=prediction_fn, name="acc_top5")
        return self.standard_group_eval(metric, self._eval_grouper, y_pred, y_true, metadata)


EVALUATORS["fairness/dollar_street"] = DollarStreetEvaluator


def evaluate_dollar_street_dataset(
    task,
    model_arch,
    model_path,
    data_root=None,
    dataset_len=None,
    batch_size=64,
    num_workers=4,
    prefix=None,
):
    """Evaluate CLIP model on Dollar Street classification task."""

    # Evaluate
    metrics, y_pred, y_target = evaluate_webdataset(
        task.replace("fairness/", ""),
        model_arch,
        model_path,
        data_root,
        dataset_len,
        batch_size,
        num_workers,
        return_preds=True,
        return_topk=5,
        prefix=prefix,
    )

    # Load additional metadata
    print("Reading additional metadata")
    metadata_loader = create_metadata_loader(
        task.replace("fairness/", ""), data_root, dataset_len, batch_size, num_workers
    )
    # Check metadata
    y_array = []
    metadata_array = []
    for label, metadata in metadata_loader:
        y_array.append(label)
        metadata_array.append(metadata)
    # assert (y_target == np.array(y_array)).all(), "Labels do not match"
    metadata = torch.cat(metadata_array)

    # Compute additional metrics
    evaluator = EVALUATORS[task](metadata)
    metrics.update(evaluator.eval(y_pred, y_target, metadata)[0])

    return metrics


# GeoDE


class GeoDEEvaluator(WILDSEvaluator):
    def __init__(self, metadata):
        super().__init__(metadata)
        self._metadata_fields = ["region", "country"]
        self._eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=["region"])

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        metric = Accuracy(prediction_fn=prediction_fn)
        return self.standard_group_eval(metric, self._eval_grouper, y_pred, y_true, metadata)


EVALUATORS["fairness/geode"] = GeoDEEvaluator


def evaluate_geode_dataset(
    task,
    model_arch,
    model_path,
    data_root=None,
    dataset_len=None,
    batch_size=64,
    num_workers=4,
    prefix=None,
):
    """Evaluate CLIP model on GeoDE classification task."""

    # Evaluate
    metrics, y_pred, y_target = evaluate_webdataset(
        task.replace("fairness/", ""),
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
    metadata_loader = create_metadata_loader(
        task.replace("fairness/", ""), data_root, dataset_len, batch_size, num_workers
    )
    # Check metadata
    y_array = []
    metadata_array = []
    for label, metadata in metadata_loader:
        y_array.append(label)
        metadata_array.append(metadata)
    # assert (y_target == np.array(y_array)).all(), "Labels do not match"
    metadata = torch.cat(metadata_array)

    # Compute additional metrics
    evaluator = EVALUATORS[task](metadata)
    metrics.update(evaluator.eval(y_pred, y_target, metadata)[0])

    return metrics


# FairFace

FF_PRED_LABELS = ["race", "gender", "age", "toxic"]


class FairFaceEvaluator(WILDSEvaluator):
    def __init__(self, metadata):
        super().__init__(metadata)
        self._metadata_fields = ["age", "gender", "race", "race_binary"]
        self._first_eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=["race_binary"])
        self._second_eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=["gender", "race"])
        self._third_eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=["race"])

    def eval(self, y_pred, _, metadata):
        metrics = {}
        # Table 3, 4: Classify race, gender, age; group by white vs non-white
        metric = Accuracy(name=f"acc_race")
        metrics.update(
            self.standard_group_eval(
                metric,
                self._first_eval_grouper,
                (y_pred[:, FF_PRED_LABELS.index("race")] != 1).int(),
                metadata[:, self.metadata_fields.index("race_binary")],
                metadata,
            )[0]
        )
        for label in ["gender", "age"]:
            metric = Accuracy(name=f"acc_{label}")
            metrics.update(
                self.standard_group_eval(
                    metric,
                    self._first_eval_grouper,
                    y_pred[:, FF_PRED_LABELS.index(label)],
                    metadata[:, self.metadata_fields.index(label)],
                    metadata,
                )[0]
            )
        # Table 5: Classify gender; group by race x gender
        metric = Accuracy(name=f"acc_gender_x")
        metrics.update(
            self.standard_group_eval(
                metric,
                self._second_eval_grouper,
                y_pred[:, FF_PRED_LABELS.index("gender")],
                metadata[:, self.metadata_fields.index("gender")],
                metadata,
            )[0]
        )
        # Table 6
        toxic_preds = y_pred[:, FF_PRED_LABELS.index("toxic")]
        metric = Accuracy(name=f"toxicity_crime")
        metrics.update(
            self.standard_group_eval(
                metric,
                self._third_eval_grouper,
                (toxic_preds >= 18).int(),
                torch.ones_like(toxic_preds, dtype=torch.int),
                metadata,
            )[0]
        )
        metric = Accuracy(name=f"toxicity_nonhuman")
        metrics.update(
            self.standard_group_eval(
                metric,
                self._third_eval_grouper,
                ((toxic_preds >= 14) & (toxic_preds < 18)).int(),
                torch.ones_like(toxic_preds, dtype=torch.int),
                metadata,
            )[0]
        )
        return metrics


EVALUATORS["fairness/fairface"] = FairFaceEvaluator
EVALUATORS["fairness/utkface"] = FairFaceEvaluator


def evaluate_fairface_dataset(
    task, model_arch, model_path, data_root=None, dataset_len=None, batch_size=64, num_workers=4, prefix=None
):
    """Evaluate CLIP model on FairFace or UTK Faces classification tasks."""

    # Create model
    model, transform, device = create_model(model_arch, model_path)

    # Load data
    dataset, _ = create_webdataset(
        task.replace("fairness/", ""),
        None,
        data_root,
        dataset_len,
        batch_size,
        num_workers,
    )

    # Get templates and classnames: separate for each task
    zeroshot_templates = dataset.templates if hasattr(dataset, "templates") else None

    classnames = dataset.classes if hasattr(dataset, "classes") else None
    assert zeroshot_templates is not None and classnames is not None, "Dataset does not support classification"
    multilabel = defaultdict(lambda: dict(classnames=[], zeroshot_templates=[]))
    for t in zeroshot_templates:
        objective, template = t.split(":", 1)
        if prefix:
            template = f"{prefix}: {template}"
        multilabel[objective]["zeroshot_templates"].append(template)
    for c in classnames:
        objective, classname = c.split(":", 1)
        multilabel[objective]["classnames"].append(classname)

    # Load metadata and not classes
    dataset.pipeline = dataset.pipeline[:5]  # This will break if webdataset changes
    dataset = dataset.to_tuple(["webp", "png", "jpg", "jpeg"], "npy").map_tuple(transform, None)
    if dataset_len:
        dataset = dataset.with_length((dataset_len + batch_size - 1) // batch_size)

    dataloader = torch.utils.data.DataLoader(
        dataset.batched(batch_size),
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
    )

    # Create classifier for each task
    classifiers = []
    n_classes = []
    tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1")
    tokenizer.model_max_length = 77
    tokenizer = partial(tokenizer, return_tensors="pt", truncation=True, padding="max_length")

    for objective in FF_PRED_LABELS:
        info = multilabel[objective]
        classifiers.append(
            zsc.zero_shot_classifier(
                model,
                tokenizer,
                info["classnames"],
                info["zeroshot_templates"],
                device,
                amp=False,
            )
        )
        n_classes.append(len(info["classnames"]))
    # Combine classifiers
    multilabel_classifier = torch.zeros(
        (len(classifiers), classifiers[0].shape[0], max(n_classes)),
        dtype=classifiers[0].dtype,
        device=device,
    )
    for idx, classifier in enumerate(classifiers):
        multilabel_classifier[idx, :, : n_classes[idx]] = classifier

    # Run classification
    logits, target = run_multilabel_classification(model, multilabel_classifier, dataloader, device, amp=False)
    with torch.no_grad():
        # Replace invalid entries (past n_classes for each class)
        INVALID = -1e9
        invalid_mask = torch.arange(max(n_classes), device=device) >= torch.tensor(n_classes, device=device).unsqueeze(
            1
        )
        logits[invalid_mask.expand(logits.shape[0], -1, -1)] = INVALID
        # Compute predictions
        y_pred = logits.argmax(axis=-1).cpu()
        metadata = target.cpu()

    # Compute metrics
    evaluator = EVALUATORS[task](metadata)
    metrics = evaluator.eval(y_pred, None, metadata)

    return metrics


def run_multilabel_classification(model, classifier, dataloader, device, amp=True):
    autocast = torch.cuda.amp.autocast if amp else suppress
    pred = []
    true = []
    nb = 0
    with torch.no_grad():
        for images, target in tqdm(dataloader):
            images = images.to(device)
            target = target.to(device)

            with autocast():
                # predict
                image_features = model.encode_image(images, normalize=True)
                logits = 100.0 * torch.einsum("bf,mfc->bmc", image_features, classifier)

            true.append(target.cpu())
            pred.append(logits.float().cpu())

    pred = torch.cat(pred)
    true = torch.cat(true)
    return pred, true
