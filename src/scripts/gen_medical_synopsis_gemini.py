import json
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

import pathlib, os, json
import logging
import requests
import random

import json
from google import genai
from google.genai import types
import time
from tqdm import tqdm
import re

client = genai.Client(api_key="YOUR_API_KEY")

# generate a concise medical summary that addresses the query

# if no medical definition / relationship available for the query
BASE_PROMPT = f"""
Given a query, write an answer to the query.

Query: [query]

"""

# if no medical relationship available for the query
MED_SUM_PROMPT_v1 = f"""
Given a query and relevant medical definitions; write an answer to the query.

Query: [query]

Definitions: 

[definitions]

"""

# main prompt
MED_SUM_PROMPT_v2 = f"""
Given a query, relevant medical definitions and relationships; write an answer to the query.

Query: [query]

Definitions: 

[definitions]

Relationships:

[relationships]

"""

COT_prompt = "Give the rationale before answering."



dataset = "trec-covid" # dataset [nfcorpus, scidocs, scifact, bioasq, trec-covid]
prompt_type = "cot" # prompting [zero_shot, cot]
model = "gemini-1.5-pro" # [gemini-2.0-flash, gemini-1.5-pro]

paraphrased_dataset = False
definitions_root_path = "llm_medical_terms/"
retriever_results_root_path = ""
llm_keywords_root_path = "gemini_out/"
q_d_relevance_root_path = "gemini_out/"




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

def generate_text(prompt, model="gemini-1.5-pro", tokenizer=None, max_new_tokens=512):
    retry_delays = [5, 20, 60]  # Delays in seconds for each retry attempt

    for attempt, delay in enumerate(retry_delays, start=1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction='You are a helpful assistant.',
                    max_output_tokens=max_new_tokens,
                    temperature=1.0,
                ),
            )
            return response.text
        except Exception as e:
            if attempt < len(retry_delays):
                time.sleep(delay)
            else:
                return ''


# load definitions
with open(definitions_root_path + f"llm_query_umls_definitions_{dataset}.json") as f:
    # Load the JSON data into a Python dictionary
    definitions = json.load(f)


# load relations
with open(definitions_root_path + f"llm_query_umls_relations_{dataset}.json") as f:
    # Load the JSON data into a Python dictionary
    relations = json.load(f)


if dataset == 'bioasq':
    data_path = '/home/bioasq'
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
else:
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = "datasets"
    data_path = util.download_and_unzip(url, out_dir)
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

dataset_tag = ''
if paraphrased_dataset:
    dataset_tag = '_paraphrased'
    # load paraphrased queries
    with open(f"queries/{dataset}_query_paraphrased_gpt4o.json") as f:
        # Load the JSON data into a Python dictionary
        queries = json.load(f)

    # update paraphrased queries to original format
    queries_p = {}
    for q in queries:
        queries_p[q] = queries[q]['query_p']

    queries = queries_p.copy()

results, retriever = get_bm25_results(corpus, queries, tag = 'phase_1', already_indexed = False)

# eval phase 1
ndcg_p1, _map_p1, recall_p1, precision_p1 = retriever.evaluate(qrels, results, retriever.k_values)
print(f"Dataset: {dataset} Method: BM25")
print(ndcg_p1, _map_p1, recall_p1, precision_p1)

medical_summaries = {} # query expansion summary
q_i = 0
for q in tqdm(queries):
    # print(queries[q])
    # print('-' * 10)
    # print(definitions[q]['definitions'])
    # print('-' * 10)
    # print(relations[q]['relations'])
    # print('+' * 10)

    if len(definitions[q]['definitions']) >= 1 and len(relations[q]['relations']) >= 1:
        defs = '\n'.join(definitions[q]['definitions'])
        rels = '\n'.join(relations[q]['relations'])
        prompt = MED_SUM_PROMPT_v2.replace('[query]', queries[q]).replace('[definitions]', defs).replace('[relationships]', rels)
    elif len(definitions[q]['definitions']) >= 1:
        defs = '\n'.join(definitions[q]['definitions'])
        rels = None
        prompt = MED_SUM_PROMPT_v1.replace('[query]', queries[q]).replace('[definitions]', defs)
    else:
        defs = None
        rels = None
        prompt = BASE_PROMPT.replace('[query]', queries[q])

    if prompt_type == "cot":
        prompt += COT_prompt
    
    if prompt != None:
        o = remove_marklied_nes(generate_text(prompt, model = model, tokenizer = None, max_new_tokens = 512))
    else:
        o = None

    medical_summaries[q] = {'query': queries[q], 'definitions': defs, 'relationships': rels, 'summary': o}

    if o == None:
        continue

    # print(prompt)
    # print('=' * 10)
    # print(o)

    queries[q] = query_expand(queries[q], 5) + query_expand(o, 1) # query expansion
    
    # if q_i == 20:
    #     break
    
    q_i += 1


# Write updated JSON back to file
with open(q_d_relevance_root_path + f"{dataset}_med_summaries_{model}_{prompt_type}{dataset_tag}.json", 'w', encoding='utf-8') as file:
    json.dump(medical_summaries, file, indent=4, ensure_ascii=False)

results, retriever = get_bm25_results(corpus, queries, tag = 'phase_1', already_indexed = True)

# eval phase 1
ndcg_p2, _map_p2, recall_p2, precision_p2 = retriever.evaluate(qrels, results, retriever.k_values)
print('=' * 50)

if paraphrased_dataset:
    print(f"Dataset: {dataset} [Paraphrased GPT 4o] Method: BM25 + Ontology-guided Summary ({model} : {prompt_type})")
else:
    print(f"Dataset: {dataset} Method: BM25 + Ontology-guided Summary ({model} : {prompt_type})")

print(ndcg_p2, _map_p2, recall_p2, precision_p2)


