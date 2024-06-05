import argparse

import numpy as np
import pandas as pd

DATASET_GROUPS = {
    "ImageNet dist. shifts": {
        "ImageNet Sketch",
        "ImageNet v2",
        "ImageNet-A",
        "ImageNet-O",
        "ImageNet-R",
        "ObjectNet",
    },
    "VTAB": {
        "Caltech-101",
        "CIFAR-100",
        "CLEVR Counts",
        "CLEVR Distance",
        "Describable Textures",
        "EuroSAT",
        "KITTI Vehicle Distance",
        "Oxford Flowers-102",
        "Oxford-IIIT Pet",
        "PatchCamelyon",
        "RESISC45",
        "SVHN",
        "SUN397",
    },
    "Retrieval": {"Flickr", "MSCOCO", "WinoGAViL"},
}


def get_aggregate_scores(results_file):
    """Returns a dictionary with aggregated scores from a results file."""
    df = pd.read_json(results_file, lines=True)
    df = pd.concat([df.drop(["metrics"], axis=1), df["metrics"].apply(pd.Series)], axis=1)
    df = df.dropna(subset=["main_metric"])
    assert len(df) == 38, f"Results file has unexpected size, {len(df)}"
    results = dict(zip(df.dataset, df.main_metric))

    aggregate_results = {"ImageNet": results["ImageNet 1k"]}

    for group, datasets in DATASET_GROUPS.items():
        score = np.mean([results[dataset] for dataset in datasets])
        aggregate_results[group] = score

    aggregate_results["Average"] = np.mean(list(results.values()))

    return aggregate_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True, help="Path to the results file.")

    args = parser.parse_args()

    scores = get_aggregate_scores(args.input)

    for group, score in scores.items():
        print(f"{group}: {score:.3f}")
