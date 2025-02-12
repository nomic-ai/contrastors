import sys

import pandas as pd
from huggingface_hub.repocard import metadata_load

TASKS = [
    "BitextMining",
    "Classification",
    "Clustering",
    "PairClassification",
    "Reranking",
    "Retrieval",
    "STS",
    "Summarization",
]

TASK_LIST_BITEXT_MINING = [
    'BUCC (de-en)',
    'BUCC (fr-en)',
    'BUCC (ru-en)',
    'BUCC (zh-en)',
    'Tatoeba (afr-eng)',
    'Tatoeba (amh-eng)',
    'Tatoeba (ang-eng)',
    'Tatoeba (ara-eng)',
    'Tatoeba (arq-eng)',
    'Tatoeba (arz-eng)',
    'Tatoeba (ast-eng)',
    'Tatoeba (awa-eng)',
    'Tatoeba (aze-eng)',
    'Tatoeba (bel-eng)',
    'Tatoeba (ben-eng)',
    'Tatoeba (ber-eng)',
    'Tatoeba (bos-eng)',
    'Tatoeba (bre-eng)',
    'Tatoeba (bul-eng)',
    'Tatoeba (cat-eng)',
    'Tatoeba (cbk-eng)',
    'Tatoeba (ceb-eng)',
    'Tatoeba (ces-eng)',
    'Tatoeba (cha-eng)',
    'Tatoeba (cmn-eng)',
    'Tatoeba (cor-eng)',
    'Tatoeba (csb-eng)',
    'Tatoeba (cym-eng)',
    'Tatoeba (dan-eng)',
    'Tatoeba (deu-eng)',
    'Tatoeba (dsb-eng)',
    'Tatoeba (dtp-eng)',
    'Tatoeba (ell-eng)',
    'Tatoeba (epo-eng)',
    'Tatoeba (est-eng)',
    'Tatoeba (eus-eng)',
    'Tatoeba (fao-eng)',
    'Tatoeba (fin-eng)',
    'Tatoeba (fra-eng)',
    'Tatoeba (fry-eng)',
    'Tatoeba (gla-eng)',
    'Tatoeba (gle-eng)',
    'Tatoeba (glg-eng)',
    'Tatoeba (gsw-eng)',
    'Tatoeba (heb-eng)',
    'Tatoeba (hin-eng)',
    'Tatoeba (hrv-eng)',
    'Tatoeba (hsb-eng)',
    'Tatoeba (hun-eng)',
    'Tatoeba (hye-eng)',
    'Tatoeba (ido-eng)',
    'Tatoeba (ile-eng)',
    'Tatoeba (ina-eng)',
    'Tatoeba (ind-eng)',
    'Tatoeba (isl-eng)',
    'Tatoeba (ita-eng)',
    'Tatoeba (jav-eng)',
    'Tatoeba (jpn-eng)',
    'Tatoeba (kab-eng)',
    'Tatoeba (kat-eng)',
    'Tatoeba (kaz-eng)',
    'Tatoeba (khm-eng)',
    'Tatoeba (kor-eng)',
    'Tatoeba (kur-eng)',
    'Tatoeba (kzj-eng)',
    'Tatoeba (lat-eng)',
    'Tatoeba (lfn-eng)',
    'Tatoeba (lit-eng)',
    'Tatoeba (lvs-eng)',
    'Tatoeba (mal-eng)',
    'Tatoeba (mar-eng)',
    'Tatoeba (max-eng)',
    'Tatoeba (mhr-eng)',
    'Tatoeba (mkd-eng)',
    'Tatoeba (mon-eng)',
    'Tatoeba (nds-eng)',
    'Tatoeba (nld-eng)',
    'Tatoeba (nno-eng)',
    'Tatoeba (nob-eng)',
    'Tatoeba (nov-eng)',
    'Tatoeba (oci-eng)',
    'Tatoeba (orv-eng)',
    'Tatoeba (pam-eng)',
    'Tatoeba (pes-eng)',
    'Tatoeba (pms-eng)',
    'Tatoeba (pol-eng)',
    'Tatoeba (por-eng)',
    'Tatoeba (ron-eng)',
    'Tatoeba (rus-eng)',
    'Tatoeba (slk-eng)',
    'Tatoeba (slv-eng)',
    'Tatoeba (spa-eng)',
    'Tatoeba (sqi-eng)',
    'Tatoeba (srp-eng)',
    'Tatoeba (swe-eng)',
    'Tatoeba (swg-eng)',
    'Tatoeba (swh-eng)',
    'Tatoeba (tam-eng)',
    'Tatoeba (tat-eng)',
    'Tatoeba (tel-eng)',
    'Tatoeba (tgl-eng)',
    'Tatoeba (tha-eng)',
    'Tatoeba (tuk-eng)',
    'Tatoeba (tur-eng)',
    'Tatoeba (tzl-eng)',
    'Tatoeba (uig-eng)',
    'Tatoeba (ukr-eng)',
    'Tatoeba (urd-eng)',
    'Tatoeba (uzb-eng)',
    'Tatoeba (vie-eng)',
    'Tatoeba (war-eng)',
    'Tatoeba (wuu-eng)',
    'Tatoeba (xho-eng)',
    'Tatoeba (yid-eng)',
    'Tatoeba (yue-eng)',
    'Tatoeba (zsm-eng)',
]

TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification (en)",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification (en)",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification (en)",
    "MassiveScenarioClassification (en)",
    "MTOPDomainClassification (en)",
    "MTOPIntentClassification (en)",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

TASK_LIST_CLUSTERING = [
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]


TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]


TASK_LIST_RERANKING = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL = [
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]

TASK_LIST_STS = [
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17 (en-en)",
    "STS22 (en)",
    "STSBenchmark",
]

TASK_LIST_SUMMARIZATION = [
    "SummEval",
]


TASK_TO_METRIC = {
    "BitextMining": "f1",
    "Clustering": "v_measure",
    "Classification": "accuracy",
    "PairClassification": "cos_sim_ap",
    "Reranking": "map",
    "Retrieval": "ndcg_at_10",
    "STS": "cos_sim_spearman",
    "Summarization": "cos_sim_spearman",
}
TASK_LIST_EN = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
    + TASK_LIST_SUMMARIZATION
)

datasets = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
    + TASK_LIST_SUMMARIZATION
)
metadata = sys.argv[1]
meta = metadata_load(metadata)

task_results = [
    sub_res
    for sub_res in meta["model-index"][0]["results"]
    if (sub_res.get("task", {}).get("type", "") in TASKS)
    and any([x in sub_res.get("dataset", {}).get("name", "") for x in datasets])
]
out = [
    {
        res["dataset"]["name"]
        .replace("MTEB ", "")
        .strip(): [
            round(score["value"], 2)
            for score in res["metrics"]
            if score["type"] == TASK_TO_METRIC.get(res["task"]["type"])
        ][0]
    }
    for res in task_results
]
out = {k: v for d in out for k, v in d.items()}
df = pd.DataFrame([out])

# df.insert(1, f"Average ({len(TASK_LIST_EN)} datasets)", df[TASK_LIST_EN].mean(axis=1, skipna=False))
# df.insert(
#     2,
#     f"Classification Average ({len(TASK_LIST_CLASSIFICATION)} datasets)",
#     df[TASK_LIST_CLASSIFICATION].mean(axis=1, skipna=False),
# )
# df.insert(
#     3, f"Clustering Average ({len(TASK_LIST_CLUSTERING)} datasets)", df[TASK_LIST_CLUSTERING].mean(axis=1, skipna=False)
# )
# df.insert(
#     4,
#     f"Pair Classification Average ({len(TASK_LIST_PAIR_CLASSIFICATION)} datasets)",
#     df[TASK_LIST_PAIR_CLASSIFICATION].mean(axis=1, skipna=False),
# )
# df.insert(
#     5, f"Reranking Average ({len(TASK_LIST_RERANKING)} datasets)", df[TASK_LIST_RERANKING].mean(axis=1, skipna=False)
# )
df.insert(
    1, f"Retrieval Average ({len(TASK_LIST_RETRIEVAL)} datasets)", df[TASK_LIST_RETRIEVAL].mean(axis=1, skipna=False)
)
# df.insert(7, f"STS Average ({len(TASK_LIST_STS)} datasets)", df[TASK_LIST_STS].mean(axis=1, skipna=False))
# df.insert(
#     8,
#     f"Summarization Average ({len(TASK_LIST_SUMMARIZATION)} dataset)",
#     df[TASK_LIST_SUMMARIZATION].mean(axis=1, skipna=False),
# )
df = df.T
df.reset_index(inplace=True)
df.columns = ["Dataset", "Score"]
print(df.to_markdown())
# write to markdown file
with open("results.md", "w") as f:
    f.write(df.to_markdown())
