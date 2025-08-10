import os
import json
import logging
from time import sleep
from tqdm import tqdm
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from pymetamap import MetaMap
import pandas as pd
import argparse

### set up logging
logging.basicConfig(level=logging.INFO)

### function to process queries using MetaMap and extract concepts
def process_queries_with_metamap(queries, metamap_base_dir, metamap_bin_dir):
    ### start metamap instance
    metam = MetaMap.get_instance(metamap_base_dir + metamap_bin_dir)
    
    ### extract concepts for each query
    processed_results = {}
    for key, query in tqdm(queries.items(), desc="Processing queries"):
        try:
            cons, errs = metam.extract_concepts([query],
                                                word_sense_disambiguation=True,
                                                # GPT 4-o
                                                # restrict_to_sts=['dsyn','sosy','fndg','anst','phsu','topp','patf','nnon','food'], # (nfcorpus)
                                                # restrict_to_sts=['dsyn','sosy','phsu','topp','bmod','neop','menp','edac','inpr','resa','cnce'], # (scidocs)
                                                restrict_to_sts=['dsyn','sosy','phsu','topp','bmod','resa','inpr','cnce','orga','patf','neop'], # (scifact) 
                                                # restrict_to_sts=['dsyn','sosy','phsu','topp','patf','bpoc','gngm','mfun','comd','bmod','resa','chem','orga','menp','cnce','evnt','inpr'] # (bioasq)
                                                # restrict_to_sts=['dsyn','sosy','phsu','topp','diap','patf','bpoc','virs','gngm','bmod','chem','popg','menp','orga','eehu','cnce','evnt'] # (trec-covid)
                                                composite_phrase=1,
                                                prune=50)
        except:
            cons = []
        
        ### store processed concepts in dictionary
        keys_of_interest = ['preferred_name', 'trigger', 'cui', 'semtypes', 'pos_info']
        concept_list = [concept._asdict() for concept in cons]
        extracted_data = [{k: concept.get(k) for k in keys_of_interest} for concept in concept_list]
        
        ### add to results dictionary
        processed_results[key] = extracted_data
    
    return processed_results

### main function
def main(dataset):
    ### pubmedqa
    if dataset == 'pubmedqa':
        # load corpus.json
        with open('pubmed_qa/corpus.json', 'r') as f:
            corpus = json.load(f)

        # load queries.json
        with open('pubmed_qa/queries.json', 'r') as f:
            queries = json.load(f)

        # load qrels.json
        with open('pubmed_qa/qrels.json', 'r') as f:
            qrels = json.load(f)
    elif dataset == 'bioasq':
        print("Not processed yet!")
    else:
        ### download and unzip dataset
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        out_dir = "datasets"
        data_path = util.download_and_unzip(url, out_dir)

        ### load dataset
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    
    ### setup metamap paths
    metamap_base_dir = '/metamap/public_mm/'
    metamap_bin_dir = 'bin/metamap20'
    metamap_pos_server_dir = 'bin/skrmedpostctl'
    metamap_wsd_server_dir = 'bin/wsdserverctl'

    # os.system(metamap_base_dir + metamap_pos_server_dir + ' start') # Part of speech tagger
    # os.system(metamap_base_dir + metamap_wsd_server_dir + ' start') # Word sense disambiguation 

    ### pause to ensure servers are up (if needed, include server start commands)
    sleep(5)
    
    ### process queries using metamap
    processed_queries = process_queries_with_metamap(queries, metamap_base_dir, metamap_bin_dir)
    
    ### save processed data to json
    output_file = f"queries_metamap/{dataset}_processed_queries.json"
    with open(output_file, 'w') as f:
        json.dump(processed_queries, f, indent=2)
    
    logging.info(f"Processed data saved to {output_file}")

### parse runtime argument
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process dataset with MetaMap')
    parser.add_argument('--dataset', required=True, help='Name of the dataset')
    args = parser.parse_args()
    
    ### run main function with provided dataset
    main(args.dataset)