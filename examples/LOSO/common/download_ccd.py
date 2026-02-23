import urllib.request
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_ccd():
    """Download and decompress the PDB Chemical Component Dictionary (CCD)."""
    url = "https://files.wwpdb.org/pub/pdb/data/monomers/components-pub.sdf.gz"
    common_dir = Path(__file__).parent
    gz_path = common_dir / "components-pub.sdf.gz"
    sdf_path = common_dir / "components-pub.sdf"

    print(f"Downloading CCD from {url}...")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=gz_path, reporthook=t.update_to)

    print(f"Decompressing {gz_path.name} to {sdf_path.name}...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(sdf_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Clean up the compressed file
    gz_path.unlink()
    print(f"Successfully downloaded and decompressed CCD to {sdf_path}")

if __name__ == "__main__":
    download_ccd()
