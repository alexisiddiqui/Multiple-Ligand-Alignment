import urllib.request
import json

def search_rcsb():
    # Example query to find PDBs for CDK2 (Uniprot P24941)
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match",
                "value": "P24941"
            }
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": 10
            }
        }
    }
    
    req = urllib.request.Request(
        "https://search.rcsb.org/rcsbsearch/v2/query?json=" + urllib.parse.quote(json.dumps(query)),
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        
    pdb_ids = [result['identifier'] for result in data.get('result_set', [])]
    print("Found PDB IDs:", pdb_ids)
    
    if not pdb_ids:
        return
        
    pdb_id = pdb_ids[0]
    
    # Now query Data API for ligands in this PDB
    data_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    req2 = urllib.request.Request(data_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req2) as resp2:
        entry_data = json.loads(resp2.read().decode())
        
    print(f"Data for {pdb_id}:")
    non_polymers = entry_data.get('rcsb_entry_info', {}).get('nonpolymer_entity_count', 0)
    print(f"Non-polymer entities: {non_polymers}")

if __name__ == "__main__":
    search_rcsb()
