"""
GASPILLOMÈTRE - Inférence et estimation du gaspillage
=======================================================
Ce script sera responsable de :
1. Charger le modèle YOLOv8 entraîné
2. Détecter les aliments sur une photo de plateau
3. Estimer le poids des restes
4. Calculer le gaspillage et sauvegarder les résultats

Pour l'instant, seul le squelette est en place.
La détection réelle n'est pas encore câblée.

Usage prévu :
    python src/inference.py --image chemin/vers/photo.jpg
    python src/inference.py --dir imageplateau/
"""
