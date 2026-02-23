import pytest
import subprocess
from pathlib import Path
from rdkit import Chem
import tempfile
import sys

@pytest.fixture
def tmp_files():
    """Create temporary query and db files for testing the CLI."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        query_path = tmp_path / "query.sdf"
        db_path = tmp_path / "db.smi"
        out_sdf = tmp_path / "out.sdf"
        out_csv = tmp_path / "out.csv"
        
        # Write Query
        mol = Chem.MolFromSmiles("c1ccccc1O")
        writer = Chem.SDWriter(str(query_path))
        writer.write(mol)
        writer.close()
        
        # Write DB
        with open(db_path, "w") as f:
            f.write("CCO\nCC(C)O\nCC(=O)O\n")
            
        yield {
            "query": str(query_path),
            "db": str(db_path),
            "out_sdf": str(out_sdf),
            "out_csv": str(out_csv)
        }

def test_cli_basic(tmp_files):
    """Test that the CLI runs successfully and produces expected files."""
    
    # We call the CLI via subprocess to ensure it parses sys.argv correctly
    result = subprocess.run([
        sys.executable, "-m", "ligand_neff.cli",
        tmp_files["query"],
        tmp_files["db"],
        "--out-sdf", tmp_files["out_sdf"],
        "--out-csv", tmp_files["out_csv"]
    ], capture_output=True, text=True)
    
    assert result.returncode == 0, f"CLI Failed: {result.stderr}"
    assert "Global Neff" in result.stdout
    assert "Global Confidence" in result.stdout
    assert "Lambda Value" in result.stdout
    
    assert Path(tmp_files["out_sdf"]).exists()
    assert Path(tmp_files["out_csv"]).exists()

def test_cli_precompute(tmp_files):
    """Test that the CLI can precompute a database and then use it."""
    
    npz_path = Path(tmp_files["query"]).parent / "test_db.npz"
    
    # Run 1: Precompute
    result1 = subprocess.run([
        sys.executable, "-m", "ligand_neff.cli",
        tmp_files["query"],
        tmp_files["db"],
        "--precompute", str(npz_path)
    ], capture_output=True, text=True)
    
    assert result1.returncode == 0, f"CLI Precompute Failed: {result1.stderr}"
    assert npz_path.exists()
    
    # Run 2: Use Precomputed
    result2 = subprocess.run([
        sys.executable, "-m", "ligand_neff.cli",
        tmp_files["query"],
        str(npz_path)
    ], capture_output=True, text=True)
    
    assert result2.returncode == 0, f"CLI with Precomputed DB Failed: {result2.stderr}"
    assert "Using precomputed .npz database cache." in result2.stdout
    assert "Global Neff" in result2.stdout 
