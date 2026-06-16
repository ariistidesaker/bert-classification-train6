import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import os

def set_seed(seed=42):
    """
    Fixe la graine aléatoire pour garantir la reproductibilité.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(preds, labels):
    """
    Calcule l'accuracy et le f1-score.
    
    Args:
        preds (list ou np.array): Prédictions du modèle.
        labels (list ou np.array): Vrais labels.
        
    Returns:
        dict: Dictionnaire contenant 'accuracy' et 'f1_score'.
    """
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'f1_score': f1
    }

def plot_confusion_matrix(preds, labels, classes=['Entailment', 'Neutral', 'Contradiction'], save_path='confusion_matrix.png'):
    """
    Génère et sauvegarde la matrice de confusion.
    """
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Prédictions')
    plt.ylabel('Vrais Labels')
    plt.title('Matrice de Confusion')
    plt.savefig(save_path)
    plt.close()

def plot_learning_curves(history, save_path='learning_curves.png'):
    """
    Génère et sauvegarde les courbes d'apprentissage (Loss et Accuracy).
    
    Args:
        history (dict): Dictionnaire contenant les listes de métriques par epoch.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Courbe de Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Courbe d'Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
