# ===============================
# MetaMap restrict_to_sts settings per dataset
# ===============================
# 
# nfcorpus:
# ['dsyn', 'sosy', 'fndg', 'anst', 'phsu', 'topp', 'patf', 'nnon', 'food']
#
# scidocs:
# ['dsyn', 'sosy', 'phsu', 'topp', 'bmod', 'neop', 'menp', 'edac', 'inpr', 'resa', 'cnce']
#
# scifact:
# ['dsyn', 'sosy', 'phsu', 'topp', 'bmod', 'resa', 'inpr', 'cnce', 'orga', 'patf', 'neop']
#
# trec-covid:
# ['dsyn', 'sosy', 'phsu', 'topp', 'diap', 'patf', 'bpoc', 'virs', 'gngm', 'bmod', 'chem', 'popg', 'menp', 'orga', 'eehu', 'cnce', 'evnt']
# ===============================

import json
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

import pathlib, os, json
import logging
import requests
import random

import json
from openai import OpenAI
from tqdm import tqdm
import re

dataset = "trec-covid" # dataset [nfcorpus, scidocs, scifact, bioasq, trec-covid]

def query_expand(q, alpha):
    if alpha == 0:
        return ' '
    out = ' '
    for _ in range(alpha):
        out += ' ' + q + ' '
    return out

# run pyserini in docker
# sudo docker run -p 7000-8000:7000-8000 -it --rm beir/pyserini-fastapi
def get_bm25_results(corpus, queries, tag = '', already_indexed = False):
    #### Convert BEIR corpus to Pyserini Format #####
    if already_indexed == False:
        pyserini_jsonl = "pyserini.jsonl"
        with open(os.path.join(data_path, pyserini_jsonl), 'w', encoding="utf-8") as fOut:
            for doc_id in corpus:
                title, text = corpus[doc_id].get("title", ""), corpus[doc_id].get("text", "")
                data = {"id": doc_id, "title": title, "contents": text}
                json.dump(data, fOut)
                fOut.write('\n')

    #### Download Docker Image beir/pyserini-fastapi ####
    #### Locally run the docker Image + FastAPI ####
    docker_beir_pyserini = "http://127.0.0.1:8000"

    #### Index documents to Pyserini #####
    index_name = f"beir/{dataset}_{tag}" # beir/scifact

    # r = requests.get(docker_beir_pyserini + "/delete/", params={"index_name": index_name})
    # print(r)

    #### Upload Multipart-encoded files ####
    if already_indexed == False:
        with open(os.path.join(data_path, "pyserini.jsonl"), "rb") as fIn:
            r = requests.post(docker_beir_pyserini + "/upload/", files={"file": fIn}, verify=False)

    if already_indexed == False:
        r = requests.get(docker_beir_pyserini + "/index/", params={"index_name": index_name})

    #### Retrieve documents from Pyserini #####
    retriever = EvaluateRetrieval()
    qids = list(queries)
    query_texts = [queries[qid] for qid in qids]
    payload = {"queries": query_texts, "qids": qids, "k": max(retriever.k_values)}

    #### Retrieve pyserini results (format of results is identical to qrels)
    results = json.loads(requests.post(docker_beir_pyserini + "/lexical/batch_search/", json=payload).text)["results"]

    return results, retriever


def remove_marklied_nes(text):
    # define the regular expression pattern for lines starting with ** something **
    pattern = r'^\*\*[^*]+\*\*'
    
    # split the text into lines
    lines = text.split('\n')
    
    # filter lines that do not start with the pattern
    filtered_lines = [line for line in lines if not re.match(pattern, line.strip())]
    
    # join the lines back into a single string
    return '\n'.join(filtered_lines)

# load definitions
with open(f"metamap_umls_definitions_{dataset}.json") as f:
    # Load the JSON data into a Python dictionary
    definitions = json.load(f)


if dataset == 'bioasq':
    data_path = '/home/bioasq'
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
else:
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = "datasets"
    data_path = util.download_and_unzip(url, out_dir)
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")


results, retriever = get_bm25_results(corpus, queries, tag = 'phase_1', already_indexed = False)

# eval phase 1
ndcg_p1, _map_p1, recall_p1, precision_p1 = retriever.evaluate(qrels, results, retriever.k_values)
print(f"Dataset: {dataset} Method: BM25")
print(ndcg_p1, _map_p1, recall_p1, precision_p1)


q_i = 0
for q in tqdm(queries):

    if len(definitions[q]['definitions']) >= 1:
        defs = '\n'.join(definitions[q]['definitions'])
    else:
        defs = ''

    queries[q] = query_expand(queries[q], 10) + query_expand(defs, 1) # query expansion
    
    q_i += 1

results, retriever = get_bm25_results(corpus, queries, tag = 'phase_1', already_indexed = True)

# eval phase 1
ndcg_p2, _map_p2, recall_p2, precision_p2 = retriever.evaluate(qrels, results, retriever.k_values)
print('=' * 50)

print(f"Dataset: {dataset} Method: BM25 + (Metamap + UMLS Definitions)")

print(ndcg_p2, _map_p2, recall_p2, precision_p2)


