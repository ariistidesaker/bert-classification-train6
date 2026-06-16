import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import os
import argparse
from sklearn.model_selection import train_test_split
from dataset import TextClassificationDataset
from model import get_model
from utils import set_seed, compute_metrics, plot_learning_curves, plot_confusion_matrix

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """
    Entraîne le modèle sur une époque.
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Déplacer les tenseurs sur le bon appareil (CPU/GPU)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Remise à zéro des gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        
        # Backward pass et optimisation
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
            
        total_loss += loss.item()
        
        # Collecte des prédictions pour les métriques
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
        
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_preds, all_labels)
    
    return avg_loss, metrics['accuracy']

def eval_epoch(model, dataloader, device):
    """
    Évalue le modèle sur le set de validation.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad(): # Désactivation du calcul des gradients
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_preds, all_labels)
    
    return avg_loss, metrics['accuracy'], metrics['f1_score'], all_preds, all_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/train.csv', help='Chemin vers le dataset CSV.')
    parser.add_argument('--model_name', type=str, default='bert-base-multilingual-cased', help='Modèle pré-entraîné.')
    parser.add_argument('--epochs', type=int, default=3, help='Nombre d\'époques.')
    parser.add_argument('--batch_size', type=int, default=16, help='Taille du batch.')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate.')
    parser.add_argument('--max_length', type=int, default=128, help='Longueur maximale des tokens.')
    parser.add_argument('--seed', type=int, default=42, help='Graine aléatoire.')
    args = parser.parse_args()

    # 1. Configuration initiale
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation de l'appareil : {device}")

    # 2. Chargement du tokenizer et du dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print("Chargement du dataset...")
    full_dataset = TextClassificationDataset(args.data_path, tokenizer, max_length=args.max_length)
    
    # 3. Split Train / Validation (80/20 stratifié)
    labels = full_dataset.data['label'].values
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)), 
        test_size=0.2, 
        stratify=labels, 
        random_state=args.seed
    )
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # 4. Chargement du modèle
    model = get_model(args.model_name, num_labels=3)
    model.to(device)

    # 5. Optimiseur et Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 6. Boucle d'entraînement
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\\n======== Epoch {epoch} / {args.epochs} ========")
        
        # Entraînement
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        # Validation
        val_loss, val_acc, val_f1, val_preds, val_labels = eval_epoch(model, val_loader, device)
        current_lr = scheduler.get_last_lr()[0] if scheduler else args.lr
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | LR: {current_lr:.2e}")
        
        # Enregistrement de l'historique
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Sauvegarde du meilleur modèle
        if val_loss < best_val_loss:
            print(f"Validation loss diminuée ({best_val_loss:.4f} --> {val_loss:.4f}). Sauvegarde du modèle...")
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            
            # Sauvegarder la matrice de confusion pour le meilleur modèle
            plot_confusion_matrix(val_preds, val_labels, save_path='confusion_matrix_best.png')

    # 7. Sauvegarde des courbes d'apprentissage
    plot_learning_curves(history, save_path='learning_curves.png')
    print("\\nEntraînement terminé ! Courbes et matrice sauvegardées.")

if __name__ == '__main__':
    main()
