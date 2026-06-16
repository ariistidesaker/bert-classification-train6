import torch
from torch.utils.data import Dataset
import pandas as pd

class TextClassificationDataset(Dataset):
    """
    Dataset PyTorch personnalisé pour la classification de texte (NLI).
    Prend en charge la tokenization, le padding et la génération d'attention masks.
    """
    def __init__(self, data_path, tokenizer, max_length=128):
        """
        Initialise le dataset.
        
        Args:
            data_path (str): Chemin vers le fichier CSV du dataset.
            tokenizer (PreTrainedTokenizer): Tokenizer Hugging Face.
            max_length (int): Longueur maximale des séquences après tokenization.
        """
        self.data = pd.read_csv(data_path)
        
        # Supprimer les lignes avec des valeurs manquantes dans les colonnes importantes
        self.data = self.data.dropna(subset=['premise', 'hypothesis', 'label'])
        
        # S'assurer que le label est bien un entier
        self.data['label'] = self.data['label'].astype(int)
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Retourne le nombre total d'exemples."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retourne un dictionnaire contenant les input_ids, attention_mask et le label
        pour un index donné.
        """
        row = self.data.iloc[idx]
        premise = str(row['premise'])
        hypothesis = str(row['hypothesis'])
        label = row['label']

        # Tokenization de la paire (premise, hypothesis)
        # Le tokenizer gérera l'ajout de [CLS] et de [SEP] entre les deux textes.
        encoding = self.tokenizer(
            premise,
            hypothesis,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
