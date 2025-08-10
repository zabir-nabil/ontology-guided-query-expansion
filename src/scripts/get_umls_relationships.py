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


def get_relations(search_key):
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

    if cui == None:
        return None

    # print(cui)

    # define function to fetch definition for a given cui
    def get_cui_relations(cui):
        ### call the api to get cui relations
        cui_relations_str = cui_api.get_relations(
            cui=cui,
            sabs="SNOMEDCT_US",
            include_relation_labels=None,
            include_additional_labels=None,
            include_obsolete=False,
            include_suppressible=False,
            page_number=1,
            page_size=25,
            save_to_file=False,
            file_path=PATH,
        )
        ### convert the string response to a python dictionary
        cui_relations = json.loads(cui_relations_str)

        ### print relations
        # logger.info("cui relations:")
        for relation in cui_relations["result"]:
            from_name = relation["relatedFromIdName"]
            to_name = relation["relatedIdName"]
            relation_label = relation["relationLabel"]
            additional_label = relation["additionalRelationLabel"]
            # print(f"{from_name} ({relation_label}) -> {to_name} [{additional_label}]")

        return cui_relations


    def get_textual_rep(relationships_data):
        # Define important relationship types and their full names for clarity
        important_relations = {
            'CHD/isa': 'has child',
            'PAR/inverse_isa': 'has parent',
            'SY/same_as': 'is synonymous with',
            'RO/associated_with': 'is associated with',
            'RO/has_associated_morphology': 'has associated morphology'
        }

        # Extract relationships from the JSON data
        results = relationships_data.get("result", [])
        hierarchical_representation = {}

        for relation in results:
            source_name = relation.get("relatedFromIdName", "Unknown")
            target_name = relation.get("relatedIdName", "Unknown")
            relation_type = relation.get("relationLabel", "Default")  # Default to "Default" if missing
            additional_label = relation.get("additionalRelationLabel", "related_to")
            full_relation_type = f"{relation_type}/{additional_label}"

            # Map the relationship type to a more descriptive term if it's important
            if full_relation_type in important_relations:
                description = important_relations[full_relation_type]
                if source_name not in hierarchical_representation:
                    hierarchical_representation[source_name] = set()
                
                # Compose the relation string with the full descriptive term
                relation_string = f"{description}: {target_name}"
                
                # Add the relation string to the set, automatically avoiding duplicates
                hierarchical_representation[source_name].add(relation_string)

        # Prepare the output text
        out = ''
        for source, relations in hierarchical_representation.items():
            out += f"{source}:\n"
            for relation in sorted(relations):  # Sorting to maintain consistent order
                out += f"  └── {relation}\n"
        
        return out

    # logic None
    time.sleep(0.1)
    rels = get_cui_relations(cui)
    rels_txt = get_textual_rep(rels)

    return rels_txt


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
        
        # Get relations for each term
        relations = []
        for term in search_terms:
            try:
                relations.append(get_relations(term))
            except:
                pass
        relations = [d for d in relations if d is not None]
        
        # Add definitions to the entry
        value["relations"] = relations

        # print(value)
        # break
    
    # Write updated JSON back to file
    with open(f"llm_medical_terms/llm_query_umls_relations_{dataset}.json", 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    dataset = 'trec-covid'
    process_json(f"llm_medical_terms/llm_medical_terms_{dataset}.json", dataset)