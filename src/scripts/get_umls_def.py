import logging
import os
import sys
import json
import time
from tqdm import tqdm
import re

from umls_python_client import UMLSClient

# Add your API key
API_KEY = "YOUR_API_KEY"

# Folder path to save files, for now this will save all the files to root
PATH = "sample_data"

search_api = UMLSClient(api_key=API_KEY).searchAPI
# initialize umls client
cui_api = UMLSClient(api_key=API_KEY).cuiAPI

def get_definitions(search_key):
    # search_key = "heart"

    search_results_exact = search_api.search(
        search_string=search_key,  # Search for the exact term
        search_type="exact",  # Use exact matching
        return_id_type="concept",  # Return CUIs
        page_number=1,  # Start from the first page
        page_size=10,  # Limit to 10 results per page
        save_to_file=False,
        file_path=PATH
    )

    search_results_exact = json.loads(search_results_exact)

    # print(search_results_exact)
    # print(type(search_results_exact))

    if len(search_results_exact['result']['results']) >= 1:
        cui = search_results_exact['result']['results'][0]['ui']
    else:
        cui = None
        # partial search results are noisy, ignoring for now
        # search_results_part = search_api.search(
        #   search_string=search_key,  # Search for the exact term
        #   partial_search=True,
        #   return_id_type="concept",  # Return CUIs
        #   page_number=1,  # Start from the first page
        #   page_size=10,  # Limit to 10 results per page
        #   save_to_file=True,
        #   file_path=PATH
        # )
        # search_results_part = json.loads(search_results_part)
        # print(search_results_part)
        # print(type(search_results_part))
        # if len(search_results_part['result']['results']) >= 1:
        #   cui = search_results_part['result']['results'][0]['ui']
        # else:
        #   cui = None

    if cui == None:
        return None

    # print(cui)

    # define function to fetch definition for a given cui
    def fetch_definition(cui):
        try:
            cui_definition = cui_api.get_definitions(
                cui=cui,
                sabs="SNOMEDCT_US",  # specifying the source abbreviation
                page_number=1,
                page_size=25,
                save_to_file=False,  # avoid saving to file directly
                file_path="",        # no file path needed
            )
            # parse json response
            definition_data = json.loads(cui_definition)
            # return definition if available
            if "result" in definition_data and definition_data["result"]:
                return definition_data["result"]
            return "N/A"
        except json.JSONDecodeError as e:
            print(f"json decoding error for cui {cui}: {e}")
            return "N/A"
        except Exception as e:
            if "Not Found" in str(e):
                print(f"resource not found for cui {cui}: {e}")
                return "Resource not found"
            print(f"error fetching definition for cui {cui}: {e}")
            return "N/A"

    # logic None
    defs = fetch_definition(cui)
    time.sleep(0.2)
    # print(defs)

    ontologies = ['MSH', 'NCI', 'SNOMEDCT_US', 'CSP']
    ontology_map = {'MSH': 'MeSH', 'NCI': 'National Cancer Institute (NCI) Thesaurus', 'SNOMEDCT_US': 'SNOMED CT', 'CSP': 'CRISP Thesaurus'} # https://www.nlm.nih.gov/research/umls/sourcereleasedocs/

    definitions = ''
    for d in defs:
        if d['rootSource'] in ontologies:
            definitions +=  d['value'] + f" (Source: {ontology_map[d['rootSource']]}) ; "

    return search_key + ': ' + definitions


# o = get_definitions('heart attack')
# print(o)

# o = get_definitions('loki')
# print(o)

def process_json(file_path, dataset):
    # Read JSON data
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    for key, value in tqdm(data.items()):
        agent_text = value.get("agent", "")
        
        # Extract search terms using regex
        search_match = re.search(r"Terms:\s*\[(.*?)\]", agent_text)
        if search_match:
            search_terms = [term.strip() for term in search_match.group(1).split(',') if term.strip()]
        else:
            search_terms = []
        
        # Get definitions for each term
        definitions = []
        for term in search_terms:
            try:
                definitions.append(get_definitions(term))
            except:
                pass
        definitions = [d for d in definitions if d is not None]
        
        # Add definitions to the entry
        value["definitions"] = definitions

        # print(value)
        # break
    
    # Write updated JSON back to file
    with open(f"llm_medical_terms/llm_query_umls_definitions_{dataset}.json", 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    dataset = 'trec-covid'
    process_json(f"llm_medical_terms/llm_medical_terms_{dataset}.json", dataset)