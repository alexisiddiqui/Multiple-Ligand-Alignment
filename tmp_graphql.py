import urllib.request
import json

def fetch_graphql():
    query = """
    {
      entries(entry_ids: ["1AQ1", "1B38", "1B39", "1E1V"]) {
        rcsb_id
        nonpolymer_entities {
          rcsb_nonpolymer_entity {
            pdbx_description
          }
          nonpolymer_comp {
            chem_comp {
              id
              type
              name
              formula_weight
            }
            rcsb_chem_comp_descriptor {
              SMILES
            }
          }
        }
      }
    }
    """
    
    req = urllib.request.Request(
        "https://data.rcsb.org/graphql",
        data=json.dumps({"query": query}).encode("utf-8"),
        headers={'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    fetch_graphql()
