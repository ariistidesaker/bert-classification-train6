import torch
import torch.nn as nn
from transformers import BertModel, AutoModelForSequenceClassification

def get_model(model_name="bert-base-multilingual-cased", num_labels=3):
    """
    Charge et retourne un modèle BERT pré-entraîné pour la classification de séquences.
    
    Args:
        model_name (str): Le nom du modèle pré-entraîné Hugging Face.
        num_labels (int): Le nombre de classes pour la classification (3 pour NLI).
        
    Returns:
        Un modèle PyTorch prêt pour le fine-tuning.
    """
    # Utilisation de AutoModelForSequenceClassification qui rajoute automatiquement 
    # la tête de classification au-dessus de BERT.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    return model
