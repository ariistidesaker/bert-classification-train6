# Devoir 3 : NLP avec PyTorch - Classification de texte avec BERT

**Nom :** Amadou Sy  
**Dataset choisi :** Contradictory, My Dear Watson (NLI Multilingue)

Ce dépôt contient l'implémentation du Devoir 3 de Deep Learning concernant le fine-tuning d'un modèle BERT pour une tâche de Natural Language Inference (NLI).

## 1. Présentation du Dataset
Le dataset utilisé contient des paires de phrases (prémisse et hypothèse) dans plusieurs langues. L'objectif est de classifier la relation entre ces deux phrases parmi trois étiquettes :
- **0 (Entailment / Déduction logique)** : L'hypothèse découle directement de la prémisse.
- **1 (Neutral / Neutre)** : L'hypothèse peut être vraie ou fausse vis-à-vis de la prémisse.
- **2 (Contradiction)** : L'hypothèse est en contradiction avec la prémisse.

**Statistiques :**
- Nombre total d'exemples : ~12 000 (dans le sous-ensemble fourni).
- Langues : Français, Anglais, Espagnol, Arabe, Chinois, Swahili, etc.
- Répartition des classes : Stratifiée lors du split (80% entraînement, 20% validation).

## 2. Description du Modèle et Choix Techniques
- **Modèle :** `bert-base-multilingual-cased`. Ce choix s'impose car le dataset contient des textes en de multiples langues. Un modèle anglais ou français unique (comme CamemBERT) serait inadapté.
- **Tokenizer :** Tokenizer associé à `bert-base-multilingual-cased`. 
- **Max Length :** Fixé à 128 tokens. La majorité des prémisses et hypothèses du dataset sont de courtes phrases, 128 tokens suffisent largement et permettent d'économiser de la VRAM.
- **Tête de classification :** 3 neurones de sortie pour les 3 classes possibles.

## 3. Instructions d'Installation et d'Exécution

### Installation des dépendances
Il est recommandé de créer un environnement virtuel puis d'installer les dépendances :
```bash
pip install -r requirements.txt
```

### Entraînement du Modèle
L'entraînement effectue une boucle manuelle en PyTorch sur les epochs spécifiées (par défaut 3) et sauvegarde le meilleur modèle (`best_model.pt`) :
```bash
python train.py --data_path data/train.csv --epochs 3 --batch_size 16 --lr 2e-5
```
*Note : Assurez-vous que le dataset est bien placé dans `data/train.csv` avant de lancer.*

### Interface Gradio
Une fois l'entraînement terminé et le fichier `best_model.pt` généré, vous pouvez lancer la démo web :
```bash
python demo.py
```
Ouvrez l'URL locale générée dans votre navigateur pour tester vos propres phrases.

## 4. Détail des Étapes et Difficultés Rencontrées
- **Tokenization des paires de phrases :** Hugging Face gère élégamment les paires de phrases en insérant automatiquement le token `[SEP]` entre la prémisse et l'hypothèse. L'attention mask est correctement généré.
- **Boucle d'entraînement manuelle :** L'absence du `Trainer` d'Hugging Face a nécessité d'implémenter nous-mêmes la gestion des tenseurs sur GPU, les passes *forward* et *backward*, le calcul de la loss et l'accumulation des prédictions pour générer les métriques.
- **Équilibrage et reproductibilité :** Nous avons forcé le split `train_test_split` de scikit-learn avec l'argument `stratify` pour s'assurer que les 3 classes soient réparties de manière égale entre l'entraînement et la validation. Le seed est fixé via une fonction utilitaire globale.

## 5. Résultats Obtenus
*Note: Les images (courbes et matrices) seront générées après exécution de `train.py` et sauvegardées dans le répertoire.*

**Métriques finales :**
- Loss (Validation) : [A compléter après exécution]
- Accuracy : [A compléter après exécution]
- F1-Score : [A compléter après exécution]

**Captures d'écran (Courbes et Matrice de Confusion) :**
*(Insérez les images `learning_curves.png` et `confusion_matrix_best.png` ici)*

## 6. Répartition du Travail
- **Amadou Sy** : Réalisation de l'intégralité du projet (traitement des données, implémentation du modèle, boucle d'entraînement PyTorch, interface Gradio et rédaction du rapport), le devoir ayant été réalisé de manière individuelle.
