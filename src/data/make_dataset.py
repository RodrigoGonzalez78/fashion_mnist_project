import os
import subprocess

def download_data():
    """Descarga el dataset de ropa usado en el libro."""
    raw_dir = "data/raw"
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
    
    # Clonar el repositorio 
    repo_url = "https://github.com/alexeygrigorev/clothing-dataset-small.git"
    subprocess.run(["git", "clone", repo_url, f"{raw_dir}/clothing-dataset-small"])
    print("Dataset descargado exitosamente.")

if __name__ == "__main__":
    download_data()