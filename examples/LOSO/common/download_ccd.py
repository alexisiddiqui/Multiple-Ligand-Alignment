import requests
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm

def download_ccd():
    """Download and decompress the PDB Chemical Component Dictionary (CCD)."""
    url = "https://files.wwpdb.org/pub/pdb/data/monomers/components-pub.sdf.gz"
    common_dir = Path(__file__).parent
    gz_path = common_dir / "components-pub.sdf.gz"
    sdf_path = common_dir / "components-pub.sdf"

    print(f"Downloading CCD from {url}...")
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(gz_path, "wb") as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

    print(f"Decompressing {gz_path.name} to {sdf_path.name}...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(sdf_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Clean up the compressed file
    gz_path.unlink()
    print(f"Successfully downloaded and decompressed CCD to {sdf_path}")

if __name__ == "__main__":
    download_ccd()
