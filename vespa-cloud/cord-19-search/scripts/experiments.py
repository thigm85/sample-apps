#! /usr/bin/env python3

import os
import json
import sys
from time import time
import numpy as np
from requests import post
from pandas import read_csv

from sentence_transformers import SentenceTransformer


def label_data(vespa_feed_file, vespa_processed_feed_file, query_relevance_file):
    with open(vespa_feed_file) as f:
        with open(vespa_processed_feed_file, "w") as feed_out:
            with open(query_relevance_file, "w") as query_rel_out:
                for line in f:
                    doc = json.loads(line)
                    title = doc["fields"]["title"]
                    abstract = doc["fields"]["abstract"]
                    if (
                        title is not None
                        and abstract is not None
                        and len(abstract) > 30
                        and not any(
                            substring in title
                            for substring in [
                                "Author index",
                                "Subject index",
                                "Contents of volume",
                                "titles contents alert",
                            ]
                        )
                    ):
                        query_rel_out.write(
                            "{}\t{}\n".format(title, doc["fields"]["id"])
                        )
                        feed_out.write(json.dumps(doc))
                        feed_out.write("\n")


def get_query_relevance_data(data_file):
    data = read_csv(data_file, sep="\t", names=["query", "relevant_id"])
    return data


def split_data(data_folder, data_file_name):
    data = get_query_relevance_data(data_file=os.path.join(data_folder, data_file_name))
    msk = np.random.rand(len(data)) < 2 / 3
    train = data[msk]
    test = data[~msk]
    train.to_csv(
        os.path.join(data_folder, "query_relevance_train.csv"),
        sep="\t",
        index=False,
        header=False,
    )
    test.to_csv(
        os.path.join(data_folder, "query_relevance_test.csv"),
        sep="\t",
        index=False,
        header=False,
    )


def retrieve_model(model_type, model_path):
    if model_type == "scibert":
        return {"model": SentenceTransformer(model_path), "model_source": "scibert"}
    else:
        raise NotImplementedError


def create_experiment_file_name(rank_profile, grammar_operator, ann, embedding, hits):
    file_name = "grammar_{}_ann_{}_rank_{}_embedding_{}_hits_{}".format(
        grammar_operator, ann, rank_profile, embedding, hits
    )
    return file_name


def create_document_embedding(text, model, model_source, normalize=True):
    if model_source == "scibert":
        vector = model.encode([text])[0].tolist()
    else:
        raise NotImplementedError
    if normalize:
        norm = np.linalg.norm(vector)
        if norm > 0.0:
            vector = vector / norm
    return vector.tolist()


def create_weakAND_operator(query, target_hits=1000):

    query_tokens = query.strip().split(" ")
    terms = ", ".join(['default contains "' + token + '"' for token in query_tokens])
    return '([{"targetNumHits": ' + str(target_hits) + "}]weakAnd(" + terms + "))"


def create_ANN_operator(ann_operator, embedding, target_hits=1000):

    ann_parameters = {"scibert": ["abstract_embedding", "vector"]}

    if ann_operator in ["abstract"]:
        return '([{{"targetNumHits": {}, "label": "nns"}}]nearestNeighbor({}, {}))'.format(
            *([target_hits] + ann_parameters[embedding])
        )
    elif ann_operator is None:
        return None
    else:
        raise ValueError("Invalid ann_operator: {}".format(ann_operator))


def create_grammar_operator(query, grammar_operator):
    if grammar_operator == "OR":
        return '([{"grammar": "any"}]userInput(@userQuery))'
    elif grammar_operator == "AND":
        return "(userInput(@userQuery))"
    elif grammar_operator == "weakAND":
        return create_weakAND_operator(query)
    elif grammar_operator is None:
        return None
    elif grammar_operator is not None:
        raise ValueError("Invalid grammar operator {}.".format(grammar_operator))


def create_yql(query, grammar_operator, ann_operator, embedding):

    operators = []
    #
    # Parse grammar operator
    #
    parsed_grammar_operator = create_grammar_operator(query, grammar_operator)
    if parsed_grammar_operator is not None:
        operators.append(parsed_grammar_operator)
    #
    # Parse ANN operator
    #
    parsed_ann_operator = create_ANN_operator(ann_operator, embedding)
    if parsed_ann_operator is not None:
        operators.append(parsed_ann_operator)

    if not operators:
        raise ValueError("Choose at least one match phase operator.")

    yql = "select * from sources * where {};".format(" or ".join(operators))

    return yql


def create_vespa_body_request(
    query,
    parsed_rank_profile,
    grammar_operator,
    ann_operator,
    embedding_type,
    hits=10,
    offset=0,
    summary=None,
    embedding_vector=None,
    tracelevel=None,
):

    body = {
        "yql": create_yql(query, grammar_operator, ann_operator, embedding_type),
        "userQuery": query,
        "hits": hits,
        "offset": offset,
        "ranking": {"profile": parsed_rank_profile, "listFeatures": "true"},
        "timeout": 1,
        "presentation.format": "json",
    }
    if tracelevel:
        body.update({"tracelevel": tracelevel})
    if summary == "minimal":
        body.update({"summary": "minimal"})
    if embedding_vector:
        if embedding_type == "scibert":
            body.update({"ranking.features.query(vector)": str(embedding_vector)})
        else:
            raise NotImplementedError

    return body


def vespa_search(vespa_url, vespa_port, body):

    r = post(vespa_url + ":" + vespa_port + "/search/", json=body)
    return r.json()


def parse_vespa_json(data):
    ranking = []
    matched_ratio = 0
    if "children" in data["root"]:
        ranking = [
            (hit["fields"]["id"], hit["relevance"])
            for hit in data["root"]["children"]
            if "fields" in hit
        ]
        matched_ratio = (
            data["root"]["fields"]["totalCount"] / data["root"]["coverage"]["documents"]
        )
    return (ranking, matched_ratio)


def evaluate(
    query_relevance,
    parsed_rank_profile,
    grammar_operator,
    ann_operator,
    embedding_type,
    vespa_url,
    vespa_port,
    hits,
    model=None,
    limit_position_count=10,
):
    rank_name = (
        str(parsed_rank_profile)
        + str(grammar_operator)
        + str(ann_operator)
        + str(embedding_type)
    )

    number_queries = 0
    total_rr = 0
    total_count = 0
    start_time = time()
    records = []
    position_count = [0] * min(hits, limit_position_count)
    matched_ratio_sum = 0
    print("{}\n".format(rank_name))
    for qid, (query, relevant_id) in query_relevance.iterrows():
        print("{}/{}\n".format(qid, query_relevance.shape[0]))
        rr = 0
        embedding_vector = None
        if model is not None:
            embedding_vector = create_document_embedding(
                text=query,
                model=model["model"],
                model_source=model["model_source"],
                normalize=True,
            )
        request_body = create_vespa_body_request(
            query=query,
            parsed_rank_profile=parsed_rank_profile,
            grammar_operator=grammar_operator,
            ann_operator=ann_operator,
            embedding_type=embedding_type,
            hits=hits,
            offset=0,
            summary="minimal",
            embedding_vector=embedding_vector,
        )
        vespa_result = vespa_search(
            vespa_url=vespa_url, vespa_port=vespa_port, body=request_body
        )
        ranking, matched_ratio = parse_vespa_json(data=vespa_result)
        matched_ratio_sum += matched_ratio
        count = 0
        for rank, hit in enumerate(ranking):
            if hit[0] == relevant_id:
                rr = 1 / (rank + 1)
                if rank < limit_position_count:
                    position_count[rank] += 1
                count += 1
        records.append({"qid": qid, "rr": rr})
        total_count += count
        total_rr += rr
        number_queries += 1
    execution_time = time() - start_time
    aggregate_metrics = {
        "rank_name": rank_name,
        "number_queries": number_queries,
        "qps": number_queries / execution_time,
        "mrr": total_rr / number_queries,
        "recall": total_count / number_queries,
        "average_matched": matched_ratio_sum / number_queries,
    }
    position_freq = [count / number_queries for count in position_count]
    return records, aggregate_metrics, position_freq


def compute_all_options(
    vespa_url,
    vespa_port,
    output_dir,
    rank_profiles,
    grammar_operators,
    ann_operators,
    embeddings,
    model_path,
    hits,
    query_relevance_file,
):
    query_relevance = get_query_relevance_data(data_file=query_relevance_file)
    for rank_profile in rank_profiles:
        for grammar_operator in grammar_operators:
            grammar_operator = None if grammar_operator is "None" else grammar_operator
            for ann in ann_operators:
                ann = None if ann is "None" else ann
                for embedding in embeddings:
                    file_name = create_experiment_file_name(
                        rank_profile, grammar_operator, ann, embedding, hits
                    )
                    file_path = os.path.join(output_dir, file_name)
                    if not os.path.exists(file_path):
                        model1 = retrieve_model(embedding, model_path)
                        try:
                            records, aggregate_metrics, position_freq = evaluate(
                                query_relevance=query_relevance,
                                parsed_rank_profile=rank_profile,
                                grammar_operator=grammar_operator,
                                ann_operator=ann,
                                embedding_type=embedding,
                                vespa_url=vespa_url,
                                vespa_port=vespa_port,
                                hits=int(hits),
                                model=model1,
                            )
                        except ValueError as e:
                            print(str(e))
                            continue
                        with open(file_path, "w") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "aggregate_metrics": aggregate_metrics,
                                        "position_freq": position_freq,
                                    }
                                )
                            )


def load_all_options(
    output_dir, rank_profiles, grammar_operators, ann_operators, embeddings, hits
):
    results = []
    for rank_profile in rank_profiles:
        for grammar_operator in grammar_operators:
            for ann in ann_operators:
                for embedding in embeddings:
                    file_name = create_experiment_file_name(
                        rank_profile, grammar_operator, ann, embedding, hits
                    )
                    file_path = os.path.join(output_dir, file_name)
                    try:
                        result = json.load(open(file_path, "r"))
                    except FileNotFoundError:
                        continue
                    result.update(
                        {
                            "rank_profile": rank_profile,
                            "grammar_operator": grammar_operator,
                            "ann_operator": ann,
                            "embedding_type": embedding,
                        }
                    )
                    results.append(result)
    return results


if __name__ == "__main__":
    QUERY_RELEVANCE_FILE = sys.argv[1]
    SCIBERT_MODEL = sys.argv[2]
    OUTPUT_DIR = sys.argv[3]

    compute_all_options(
        vespa_url="http://localhost",
        vespa_port="8080",
        output_dir=OUTPUT_DIR,
        rank_profiles=["default", "bm25", "scibert", "bm25_scibert"],
        grammar_operators=[None, "OR", "AND", "weakAND"],
        ann_operators=[None, "abstract"],
        embeddings=["scibert"],
        model_path=SCIBERT_MODEL,
        hits=100,
        query_relevance_file=QUERY_RELEVANCE_FILE,
    )
