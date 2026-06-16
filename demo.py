import gradio as gr
import torch
from transformers import AutoTokenizer
from model import get_model

# Paramètres globaux
MODEL_NAME = "bert-base-multilingual-cased"
MODEL_PATH = "best_model.pt"

# Charger le tokenizer et le modèle (singleton pour l'interface Gradio)
print("Chargement du tokenizer et du modèle...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = get_model(MODEL_NAME, num_labels=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    print("Modèle chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    # En cas d'absence de best_model.pt, l'interface plantera à l'inférence.
    # Ceci est normal car on attend que l'utilisateur entraîne le modèle d'abord.

# Définition des labels
LABELS = ["Entailment (Déduction logique)", "Neutral (Neutre)", "Contradiction"]

def predict(premise, hypothesis):
    """
    Fonction de prédiction pour Gradio.
    """
    if not premise or not hypothesis:
        return "Veuillez entrer une prémisse et une hypothèse.", {}
        
    # Tokenization
    inputs = tokenizer(
        premise,
        hypothesis,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Inférence
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
        
    # Formatage de la sortie
    predicted_class_id = torch.argmax(logits, dim=1).item()
    predicted_label = LABELS[predicted_class_id]
    
    # Dictionnaire des probabilités pour l'élément Label de Gradio
    prob_dict = {LABELS[i]: prob for i, prob in enumerate(probabilities)}
    
    return predicted_label, prob_dict

# Interface Gradio
title = "Démonstration NLI - BERT Classification"
description = """
Cette interface permet de tester notre modèle de **Natural Language Inference (NLI)** fine-tuné avec BERT.
Saisissez une **Prémisse** (un fait de base) et une **Hypothèse** (une affirmation basée sur la prémisse), 
puis le modèle prévoira la relation entre elles :
- **Entailment** : L'hypothèse est déduite de la prémisse.
- **Neutral** : L'hypothèse n'a pas de rapport de contradiction ni de déduction directe avec la prémisse.
- **Contradiction** : L'hypothèse contredit la prémisse.
"""

examples = [
    ["Les enfants jouent dans le parc avec un ballon rouge.", "Des jeunes s'amusent avec un jouet rouge à l'extérieur."],
    ["Le président a annoncé une nouvelle loi sur l'environnement.", "Le gouvernement refuse de prendre des mesures écologiques."],
    ["A man is playing a guitar on stage.", "A woman is singing a song."]
]

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(lines=2, label="Prémisse", placeholder="Entrez la phrase de prémisse ici..."),
        gr.Textbox(lines=2, label="Hypothèse", placeholder="Entrez la phrase d'hypothèse ici...")
    ],
    outputs=[
        gr.Textbox(label="Classe Prédite"),
        gr.Label(label="Probabilités", num_top_classes=3)
    ],
    title=title,
    description=description,
    examples=examples,
    theme="default"
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
