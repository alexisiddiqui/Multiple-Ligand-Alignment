import argparse

def main():
    parser = argparse.ArgumentParser(description="Ligand Neff Computation")
    parser.add_argument("query", help="Path to query molecule")
    parser.add_argument("database", help="Path to reference database")
    args = parser.parse_args()
    print("Neff computation tool.")

if __name__ == "__main__":
    main()
