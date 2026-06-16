import gdown
import os

def download_dataset():
    """
    Télécharge le dataset depuis Google Drive si ce n'est pas déjà fait.
    """
    file_id = '1pOBv2SZUv_KYV8qJF-JXOwhWi-4HQxMy'
    output = 'data/train.csv'
    
    if not os.path.exists(output):
        print("Téléchargement du dataset...")
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output, quiet=False)
        print("Téléchargement terminé.")
    else:
        print("Le dataset est déjà présent dans data/train.csv.")

if __name__ == '__main__':
    # S'assurer que le script est exécuté depuis la racine du repo
    if not os.path.exists('data'):
        os.makedirs('data')
    download_dataset()
