import ast
import random

import pandas as pd
from datasets import concatenate_datasets, load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

###############################################


def load_loco_dataset(dataset_name: str, split: str, document_column: str, query_column: str, subset=None):

    if subset is None:
        dataset = load_dataset(dataset_name)[split]
    else:
        dataset = load_dataset(dataset_name, subset)[split]

    dataset = dataset[dataset["final_decision"].apply(lambda x: x == "yes")]

    corpus = {}
    queries = {}
    qrels = {}
    for row in tqdm(range(len(dataset))):
        corpus["Passage_" + str(row)] = {"title": "", "text": dataset[row][document_column]}
        queries['Query_' + str(row)] = dataset[row][query_column]
        qrels['Query_' + str(row)] = {"Passage_" + str(row): 1}

    print("Example Query")
    print(list(queries.values())[10])
    print("Example Passage (cutoff at 200 characters)")
    print(list(corpus.values())[10]['text'][:200])

    return corpus, queries, qrels


###############################################


def load_multi_news(dataset_name: str, split: str, document_column: str, query_column: str, subset=None):

    if subset is None:
        dataset = load_dataset(dataset_name)[split]
    else:
        dataset = load_dataset(dataset_name, subset)[split]

    corpus = {}
    queries = {}
    qrels = {}
    for row in tqdm(range(len(dataset))):
        corpus["Passage_" + str(row)] = {"title": "", "text": dataset[row][document_column]}
        queries['Query_' + str(row)] = dataset[row][query_column]
        qrels['Query_' + str(row)] = {"Passage_" + str(row): 1}

    print("Example Query")
    print(list(queries.values())[10])
    print("Example Passage (cutoff at 200 characters)")
    print(list(corpus.values())[10]['text'][:200])

    return corpus, queries, qrels


###############################################


def load_pubmed_qa(dataset_name: str, split: str, document_column: str, query_column: str, subset=None):

    if subset is None:
        dataset = load_dataset(dataset_name)[split]
    else:
        dataset = load_dataset(dataset_name, subset)[split]

    # Filter out for successful query-document-answer triples
    dataset = dataset.filter(lambda example: example["final_decision"] == "yes")

    dataset = dataset.to_pandas()
    train, test = train_test_split(dataset, test_size=0.1, random_state=42)
    if split == "train":
        dataset = train
    elif split == "test":
        dataset = test
    else:
        raise ValueError("No data split selected!")

    ########################

    corpus = {}
    queries = {}
    qrels = {}
    for row in tqdm(range(len(dataset))):
        assert type(dataset[row][document_column]['contexts']) == list
        corpus["Passage_" + str(row)] = {
            "title": "",
            "text": (" ").join(dataset.iloc[row][document_column]['contexts']),
        }
        queries['Query_' + str(row)] = dataset.iloc[row][query_column]
        qrels['Query_' + str(row)] = {"Passage_" + str(row): 1}

    print("Example Query")
    print(list(queries.values())[10])
    print("Example Passage (cutoff at 200 characters)")
    print(list(corpus.values())[10]['text'][:200])

    return corpus, queries, qrels


###############################################


def load_tau_fs(dataset_name: str, split: str, document_column: str, query_column: str, subset=None):

    if subset is None:
        dataset = load_dataset(dataset_name)[split]
    else:
        dataset = load_dataset(dataset_name, subset)[split]

    corpus = {}
    queries = {}
    qrels = {}
    for row in tqdm(range(len(dataset))):
        corpus["Passage_" + str(row)] = {"title": "", "text": dataset[row][document_column]}
        queries['Query_' + str(row)] = dataset[row][query_column]
        qrels['Query_' + str(row)] = {"Passage_" + str(row): 1}

    print("Example Query")
    print(list(queries.values())[10])
    print("Example Passage (cutoff at 200 characters)")
    print(list(corpus.values())[10]['text'][:200])

    return corpus, queries, qrels


###############################################


def load_tau_scrolls_for_summ_screen_fd_gov_report_qmsum(
    dataset_name: str, split: str, document_column: str, query_column: str, subset=None
):

    if split not in ["train", "validation"]:
        ValueError(f"No data split selected! Subset: {subset}, Split: {split}")

    if subset is None:
        dataset = load_dataset(dataset_name)[split]
    else:
        dataset = load_dataset(dataset_name, subset)[split]

    corpus = {}
    queries = {}
    qrels = {}

    for row in tqdm(range(len(dataset))):
        corpus["Passage_" + str(row)] = {"title": "", "text": dataset[row][document_column]}
        queries['Query_' + str(row)] = dataset[row][query_column]
        qrels['Query_' + str(row)] = {"Passage_" + str(row): 1}

    print("Example Query")
    print(list(queries.values())[10])
    print("Example Passage (cutoff at 200 characters)")
    print(list(corpus.values())[10]['text'][:200])

    return corpus, queries, qrels


###############################################


def load_qasper(dataset_name: str, split: str, document_column: str, query_column: str, subset=None):

    if subset is None:
        dataset = load_dataset(dataset_name)[split]
    else:
        dataset = load_dataset(dataset_name, subset)[split]

    assert query_column != "full_text"

    corpus = {}
    queries = {}
    qrels = {}
    for row in tqdm(range(len(dataset))):
        if document_column == "full_text":
            passage_text = ""
            paragraphs = [
                " ".join(paragraph_sublist) for paragraph_sublist in dataset[row][document_column]['paragraphs']
            ]
            assert len(dataset[row][document_column]['section_name']) == len(paragraphs)
            for section_header, paragraph in zip(dataset[row][document_column]['section_name'], paragraphs):
                if section_header is not None or paragraph is not None:
                    if section_header is None:
                        passage_text += paragraph + "\n"
                    elif paragraph is None:
                        passage_text += section_header + "\n"
                    else:
                        passage_text += section_header + "\n" + paragraph + "\n"
            passage_text = passage_text.strip()
            corpus["Passage_" + str(row)] = {"title": "", "text": passage_text}
        else:
            corpus["Passage_" + str(row)] = {"title": "", "text": dataset[row][document_column]}

        queries['Query_' + str(row)] = dataset[row][query_column]
        qrels['Query_' + str(row)] = {"Passage_" + str(row): 1}

    print("Example Query")
    print(list(queries.values())[10])
    print("Example Passage (cutoff at 200 characters)")
    print(list(corpus.values())[10]['text'][:200])

    return corpus, queries, qrels


###############################################


def load_trivia_qa(dataset_name: str, split: str, document_column: str, query_column: str, subset=None):

    if subset is None:
        dataset = load_dataset(dataset_name)[split]
    else:
        dataset = load_dataset(dataset_name, subset)[split]

    corpus = {}
    queries = {}
    qrels = {}
    for row in tqdm(range(len(dataset))):
        corpus["Passage_" + str(row)] = {"title": "", "text": dataset[row][document_column]}
        queries['Query_' + str(row)] = dataset[row][query_column]
        qrels['Query_' + str(row)] = {"Passage_" + str(row): 1}

    print("Example Query")
    print(list(queries.values())[10])
    print("Example Passage (cutoff at 200 characters)")
    print(list(corpus.values())[10]['text'][:200])

    return corpus, queries, qrels


###############################################


def load_kilt_dataset(dataset_name: str, split: str, document_column: str, query_column: str, subset=None):

    if split not in ["train", "validation"]:
        ValueError(f"No data split selected! Subset: {subset}, Split: {split}")

    folder_path = "kilt_datasets/" + subset + "/"
    file_path = folder_path + subset + "_reformatted_full_articles_True_shuffle_sections_True_" + split + ".tsv"
    # file_path = folder_path + subset + "_reformatted_full_articles_True_shuffle_sections_False_gather_distractor_documents_set_True_distractor_documents_count_9_" + split + ".tsv"
    dataset = pd.read_csv(file_path, sep="\t")

    print("Size of dataset before filtering: " + str(len(dataset)))
    dataset = dataset[dataset[document_column].apply(lambda x: not isinstance(x, float))]
    dataset = dataset[dataset[query_column].apply(lambda x: not isinstance(x, float))]
    print("Size of dataset after filtering: " + str(len(dataset)))

    wiki_id_to_passage_id = {}
    corpus = {}
    queries = {}
    qrels = {}
    for row in tqdm(range(len(dataset))):
        if subset == "hotpotqa":
            wiki_ids = ast.literal_eval(dataset.iloc[row]['wikipedia_id'])
            assert type(wiki_ids) == list
            qrels['Query_' + str(row)] = {}
            for wiki_id, passage_sub_id in zip(wiki_ids, range(len(wiki_ids))):
                if wiki_id not in wiki_id_to_passage_id:
                    new_passage_id = "Passage_" + str(row) + "_" + str(passage_sub_id)
                    wiki_id_to_passage_id[wiki_id] = new_passage_id
                    corpus[new_passage_id] = {"title": "", "text": dataset.iloc[row][document_column]}
                    qrels['Query_' + str(row)][new_passage_id] = 1
                else:
                    new_passage_id = wiki_id_to_passage_id[wiki_id]
                    qrels['Query_' + str(row)][new_passage_id] = 1

            queries['Query_' + str(row)] = dataset.iloc[row][query_column]

        else:
            wiki_id = dataset.iloc[row]["wikipedia_id"]
            if wiki_id not in wiki_id_to_passage_id:
                new_passage_id = "Passage_" + str(row)
                corpus[new_passage_id] = {"title": "", "text": dataset.iloc[row][document_column]}
                queries['Query_' + str(row)] = dataset.iloc[row][query_column]
                qrels['Query_' + str(row)] = {new_passage_id: 1}
                wiki_id_to_passage_id[wiki_id] = new_passage_id
            else:
                new_passage_id = wiki_id_to_passage_id[wiki_id]
                queries['Query_' + str(row)] = dataset.iloc[row][query_column]
                qrels['Query_' + str(row)] = {new_passage_id: 1}

    print("Example Query")
    print(list(queries.values())[10])
    print("Example Passage (cutoff at 200 characters)")
    print(list(corpus.values())[10]['text'][:200])

    return corpus, queries, qrels


###############################################


def load_kilt_dataset_for_needle_in_haystack(
    dataset_name: str, split: str, document_column: str, query_column: str, subset=None
):

    if split not in ["train", "validation"]:
        ValueError(f"No data split selected! Subset: {subset}, Split: {split}")

    folder_path = "kilt_datasets/" + subset + "/"
    # file_path = folder_path + subset + "_reformatted_full_articles_True_shuffle_sections_True_" + split + ".tsv"
    file_path = (
        folder_path
        + subset
        + "_reformatted_full_articles_True_shuffle_sections_False_gather_distractor_documents_set_True_distractor_documents_count_100_"
        + split
        + ".tsv"
    )
    # file_path = folder_path + subset + "_reformatted_full_articles_True_shuffle_sections_False_gather_distractor_documents_set_True_distractor_documents_count_9_" + split + ".tsv"
    dataset = pd.read_csv(file_path, sep="\t")

    print("Size of dataset before filtering: " + str(len(dataset)))
    dataset = dataset[dataset[document_column].apply(lambda x: not isinstance(x, float))]
    dataset = dataset[dataset[query_column].apply(lambda x: not isinstance(x, float))]
    print("Size of dataset after filtering: " + str(len(dataset)))

    wiki_id_to_passage_id = {}
    corpus = {}
    queries = {}
    qrels = {}
    for row in tqdm(range(len(dataset))):
        if subset == "hotpotqa":
            wiki_ids = ast.literal_eval(dataset.iloc[row]['wikipedia_id'])
            assert type(wiki_ids) == list
            qrels['Query_' + str(row)] = {}
            for wiki_id, passage_sub_id in zip(wiki_ids, range(len(wiki_ids))):
                if wiki_id not in wiki_id_to_passage_id:
                    new_passage_id = "Passage_" + str(row) + "_" + str(passage_sub_id)
                    wiki_id_to_passage_id[wiki_id] = new_passage_id
                    corpus[new_passage_id] = {"title": "", "text": dataset.iloc[row][document_column]}
                    qrels['Query_' + str(row)][new_passage_id] = 1
                else:
                    new_passage_id = wiki_id_to_passage_id[wiki_id]
                    qrels['Query_' + str(row)][new_passage_id] = 1

            queries['Query_' + str(row)] = dataset.iloc[row][query_column]

        else:
            wiki_id = dataset.iloc[row]["wikipedia_id"]
            if wiki_id not in wiki_id_to_passage_id:
                new_passage_id = "Passage_" + str(row)
                corpus[new_passage_id] = {"title": "", "text": dataset.iloc[row][document_column]}
                queries['Query_' + str(row)] = dataset.iloc[row][query_column]
                qrels['Query_' + str(row)] = {new_passage_id: 1}
                wiki_id_to_passage_id[wiki_id] = new_passage_id
            else:
                new_passage_id = wiki_id_to_passage_id[wiki_id]
                queries['Query_' + str(row)] = dataset.iloc[row][query_column]
                qrels['Query_' + str(row)] = {new_passage_id: 1}

    print("Example Query")
    print(list(queries.values())[10])
    print("Example Passage (cutoff at 200 characters)")
    print(list(corpus.values())[10]['text'][:200])

    return corpus, queries, qrels


###############################################


def load_long_bench(dataset_name: str, split: str, document_column: str, query_column: str, subset=None):

    # datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", "trec",
    #            "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

    if subset == "total":
        datasets = [
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "gov_report",
            "multi_news",
            "passage_retrieval_en",
        ]  # "lcc", "repobench-p"
    else:
        datasets = [subset]

    total_loaded_datasets = []
    for dataset in datasets:
        if dataset == "vlsp":
            loaded_dataset = load_dataset("ghomasHudson/muld", "VLSP", split="test")
            assert type(loaded_dataset['output'][0]) == list
            new_outputs = [loaded_dataset['output'][row][0] for row in range(len(loaded_dataset))]
            assert type(new_outputs[0]) == str
            loaded_dataset = loaded_dataset.remove_columns("output")
            loaded_dataset = loaded_dataset.add_column("output", new_outputs)
        else:
            loaded_dataset = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')

        if dataset in ["gov_report", "multi_news"]:  # reformat input for gov_report
            assert type(loaded_dataset['answers'][0]) == list
            new_inputs = [loaded_dataset['answers'][row][0] for row in range(len(loaded_dataset))]
            assert type(new_inputs[0]) == str
            loaded_dataset = loaded_dataset.remove_columns("input")
            loaded_dataset = loaded_dataset.add_column("input", new_inputs)
        total_loaded_datasets.append(loaded_dataset)

    combined_dataset = concatenate_datasets(total_loaded_datasets)

    ########################################

    combined_dataset = combined_dataset.to_pandas()
    train, test = train_test_split(combined_dataset, test_size=0.2, random_state=42)
    if split == "train":
        combined_dataset = train
    elif split == "test":
        combined_dataset = test
    else:
        raise ValueError(f"No data split selected! Subset: {subset}, Split: {split}")

    ########################################

    corpus = {}
    queries = {}
    qrels = {}
    for row in tqdm(range(len(combined_dataset))):
        corpus["Passage_" + str(row)] = {"title": "", "text": combined_dataset.iloc[row][document_column]}
        queries['Query_' + str(row)] = combined_dataset.iloc[row][query_column]
        qrels['Query_' + str(row)] = {"Passage_" + str(row): 1}

    print("Example Query")
    print(list(queries.values())[10])
    print("Example Passage (cutoff at 200 characters)")
    print(list(corpus.values())[10]['text'][:200])
    # print("Size of LongBench: " + str(len(corpus)))

    return corpus, queries, qrels


###############################################


def load_tau_scrolls_needle(
    dataset_name: str, split: str, document_column: str, query_column: str, subset=None, answer_position=0
):

    if split not in ["train", "validation"]:
        ValueError(f"No data split selected! Subset: {subset}, Split: {split}")

    if subset is None:
        dataset = load_dataset("tau/scrolls")[split]
    else:
        dataset = load_dataset("tau/scrolls", subset)[split]

    corpus = {}
    queries = {}
    qrels = {}

    for row in tqdm(range(len(dataset))):
        corpus["Passage_" + str(row)] = {"title": "", "text": dataset[row][document_column]}
        queries['Query_' + str(row)] = dataset[row][query_column]
        qrels['Query_' + str(row)] = {"Passage_" + str(row): 1}

    ########################################################

    new_corpus = {}
    new_queries = {}
    new_qrels = {}

    test_set_rows_cutoff = 300

    for row in tqdm(range(len(dataset))):
        answer_text = corpus["Passage_" + str(row)]['text']
        if row < test_set_rows_cutoff:
            new_queries['Query_' + str(row)] = queries['Query_' + str(row)]
            new_qrels['Query_' + str(row)] = qrels['Query_' + str(row)]

        distractor_passages_rows = [random.randint(test_set_rows_cutoff, len(dataset) - 1) for _ in range(1)]
        distractor_passages = [dataset[row_int][document_column] for row_int in distractor_passages_rows]
        distractor_passages.insert(answer_position, answer_text)
        new_corpus["Passage_" + str(row)] = {"title": "", "text": (' ').join(distractor_passages)}

    print("Example Query")
    print(list(new_queries.values())[10])
    print("Example Passage (cutoff at 200 characters)")
    print(list(new_corpus.values())[10]['text'][:200])

    return new_corpus, new_queries, new_qrels
