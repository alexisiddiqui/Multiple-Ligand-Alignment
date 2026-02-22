import argparse
import sys
from pathlib import Path
from ligand_neff.config import load_config, NeffConfig
from ligand_neff.io.query import load_query
from ligand_neff.io.database import load_database
from ligand_neff.compute import compute_neff

def main():
    parser = argparse.ArgumentParser(description="Multiple Ligand Alignment (MLA) - Neff Computation")
    parser.add_argument("query", help="Path to query molecule (SDF)")
    parser.add_argument("database", help="Path to reference database (SDF, SMILES, or .npz)")
    parser.add_argument("--config", "-c", help="Path to YAML configuration file (optional)", default=None)
    parser.add_argument("--precompute", help="If provided and database is SDF/SMILES, save out an .npz file", default=None)
    parser.add_argument("--out-sdf", help="Path to save output query SDF with Neff properties", default=None)
    parser.add_argument("--out-csv", help="Path to save per-atom breakdown CSV", default=None)
    parser.add_argument("--plot", help="Path to save 2D molecule depiction (.png)", default=None)
    parser.add_argument("--plot-breakdown", help="Path to save bar chart breakdown (.png)", default=None)
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        print(f"Loading configuration from {args.config}...")
        config = load_config(args.config)
    else:
        print("Using default configuration.")
        config = NeffConfig()
        
    try:
        # Load molecules
        print(f"Loading query from {args.query}...")
        query_mol = load_query(args.query)
        
        print(f"Loading database from {args.database}...")
        db_mols = None
        precomputed_db = None
        
        if args.database.endswith('.npz'):
            precomputed_db = args.database
            print("Using precomputed .npz database cache.")
        else:
            db_mols = load_database(args.database)
            print(f"Loaded {len(db_mols)} raw reference molecules.")
            
            if args.precompute:
                from ligand_neff.io.database import precompute_database
                print(f"Precomputing and saving to {args.precompute}...")
                precompute_database(args.database, args.precompute, config)
                precomputed_db = args.precompute
        
        # Run calculation
        print("Computing Neff scores...")
        result = compute_neff(query_mol, db_mols=db_mols, config=config, precomputed_db=precomputed_db)
        
        print("\n--- Results ---")
        print(f"Global Neff:       {result.global_neff:.4f}")
        print(f"Global Confidence:  {result.global_confidence:.4f}")
        print(f"Lambda Value:       {result.lambda_value:.4f}")
        print(f"References Used:    {result.n_references_used}")
        print("---------------")
        
        # Handle outputs
        if args.out_sdf:
            print(f"Saving SDF to {args.out_sdf}...")
            result.to_sdf(args.out_sdf)
            
        if args.out_csv:
            print(f"Saving CSV to {args.out_csv}...")
            result.to_csv(args.out_csv)
            
        if args.plot:
            print(f"Saving 2D plot to {args.plot}...")
            img = result.plot()
            img.save(args.plot)
            
        if args.plot_breakdown:
            print(f"Saving chart to {args.plot_breakdown}...")
            fig = result.plot_breakdown()
            fig.savefig(args.plot_breakdown)
            
        print("Done.")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
