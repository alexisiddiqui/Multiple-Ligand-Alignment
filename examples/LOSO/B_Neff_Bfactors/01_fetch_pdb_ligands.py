import urllib.request
import json
import time
import pickle
from pathlib import Path
from rdkit import Chem

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# Exclude lists for biological cofactors and crystallization buffers/salts
EXCLUDED_COMPONENTS = {
    "HOH", "SO4", "CL", "GOL", "EDO", "MG", "ZN", "PO4", "DTT", "FMT", "ACT",
    "BME", "PEG", "PGE", "PG4", "1PE", "NHE", "TRS", "MES", "HEZ", "EPE",
    "ATP", "ADP", "AMP", "GTP", "GDP", "GMP", "NAD", "NADH", "NADP", "FAD"
}

def search_rcsb_cdk2(limit: int = 100) -> list[str]:
    """Search for CDK2 PDB IDs."""
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
            "paginate": {"start": 0, "rows": limit}
        }
    }
    
    print(f"Searching RCSB for {limit} CDK2 PDBs...")
    req = urllib.request.Request(
        "https://search.rcsb.org/rcsbsearch/v2/query?json=" + urllib.parse.quote(json.dumps(query)),
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        
    pdb_ids = [result['identifier'] for result in data.get('result_set', [])]
    return pdb_ids

def find_ligands_in_pdbs(pdb_ids: list[str]) -> dict:
    """Use GraphQL to find valid non-polymer drug-like ligands for the given PDB IDs."""
    query = """
    {
      entries(entry_ids: [%s]) {
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
          }
        }
      }
    }
    """ % ", ".join(f'"{pid}"' for pid in pdb_ids)
    
    print(f"Querying graphQL for non-polymers in '{', '.join(pdb_ids)}'...")
    req = urllib.request.Request(
        "https://data.rcsb.org/graphql",
        data=json.dumps({"query": query}).encode("utf-8"),
        headers={'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    
    pdb_to_ligands = {}
    for entry in data.get("data", {}).get("entries", []):
        pdb_id = entry["rcsb_id"]
        valid_ligands = []
        
        nonpolymer_entities = entry.get("nonpolymer_entities")
        if not nonpolymer_entities:
            continue
            
        for entity in nonpolymer_entities:
            comp = entity.get("nonpolymer_comp", {}).get("chem_comp", {})
            comp_id = comp.get("id")
            weight = comp.get("formula_weight", 0)
            
            # Simple heuristic: Not in exclude list, weight > 150 Da to exclude small ions/buffers
            if comp_id and comp_id not in EXCLUDED_COMPONENTS and weight > 150:
                valid_ligands.append(comp_id)
                
        if valid_ligands:
            pdb_to_ligands[pdb_id] = valid_ligands
            
    return pdb_to_ligands

def extract_ligand_pdb_block(pdb_text: str, comp_ids: list[str]) -> dict:
    """Extract only the HETATM lines for the specific comp_id."""
    # Sometimes a PDB might have multiple valid ligands, we'll take the first one found in the structure
    ligand_blocks = {}
    
    for comp_id in comp_ids:
        block_lines = []
        for line in pdb_text.splitlines():
            # HETATM records
            if line.startswith("HETATM"):
                # The resName is in columns 18-20
                res_name = line[17:20].strip()
                if res_name == comp_id:
                    block_lines.append(line)
        if block_lines:
            ligand_blocks[comp_id] = "\n".join(block_lines)
            
    return ligand_blocks

def fetch_pdb_ligands():
    """Main execution to fetch and process PDBs."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    pdb_ids = search_rcsb_cdk2(limit=500)
    if not pdb_ids:
        print("No PDBs found.")
        return
        
    pdb_to_ligands = find_ligands_in_pdbs(pdb_ids)
    
    print(f"\nIdentified target ligands for {len(pdb_to_ligands)} PDBs:")
    for pdb_id, ligands in pdb_to_ligands.items():
        print(f"  {pdb_id}: {', '.join(ligands)}")
        
    results = []
    
    for pdb_id, targets in pdb_to_ligands.items():
        print(f"Downloading PDB structure for {pdb_id}...")
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            with urllib.request.urlopen(url) as response:
                pdb_text = response.read().decode("utf-8")
        except Exception as e:
            print(f"  Failed: {e}")
            continue
            
        blocks = extract_ligand_pdb_block(pdb_text, targets)
        for comp_id, block in blocks.items():
            # Ask RDKit to parse the HETATM block and infer bonds purely from 3D coords
            mol = Chem.MolFromPDBBlock(block, proximityBonding=True, removeHs=True)
            if mol is None:
                print(f"  Failed to parse block for {pdb_id}:{comp_id} with RDKit.")
                continue
            
            # Extract atom B-factors
            b_factors = []
            valid = True
            for atom in mol.GetAtoms():
                mi = atom.GetMonomerInfo()
                if mi is None:
                    valid = False
                    break
                b_factors.append(mi.GetTempFactor())
                
            if not valid or not b_factors:
                print(f"  Failed to extract B-factors for {pdb_id}:{comp_id}")
                continue
                
            results.append({
                "pdb_id": pdb_id,
                "ligand_id": comp_id,
                "smiles": Chem.MolToSmiles(mol),
                "mol": mol,
                "b_factors": b_factors
            })
            print(f"  Successfully extracted {pdb_id}:{comp_id} ({len(b_factors)} atoms)")
            # Take only the first valid ligand to keep it 1:1 for the example list
            break
            
        time.sleep(0.5)
        
    print(f"\nTotal valid extracted ligands: {len(results)}")
    out_file = DATA_DIR / "pdb_ligands.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    fetch_pdb_ligands()
