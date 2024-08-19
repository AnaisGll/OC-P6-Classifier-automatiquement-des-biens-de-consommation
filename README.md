# OC-P6-Classifier-automatiquement-des-biens-de-consommation

PLace de Marché, une entreprise anglophone, souhaite lancer une marketplace e-commerce. Pour l'instant, l'attribution de la catégorie d'un article est effectuée manuellement par les vendeurs, et est donc peu fiable. De plus, le volume des articles est pour l’instant très petit. Il devient nécessaire d'automatiser cette tâche d‘attribution de la catégorie pour une meilleure expérience utilisateur des vendeurs et acheteurs. 

# Missions 

1) Réaliser, dans une première itération, une étude de faisabilité d'un moteur de classification d'articles, basé sur une image et une description, pour l'automatisation de l'attribution de la catégorie de l'article.

2) Analyser les descriptions textuelles et les images des produits, au travers :
    - Un prétraitement des données texte ou image suivant le cas 
    - Une extraction de features 
    - Une réduction en 2 dimensions, afin de projeter les produits sur un graphique 2D, sous la forme de points dont la couleur correspondra à la catégorie réelle 
    - Analyse du graphique afin d’en déduire ou pas, à l’aide des descriptions ou des images, la faisabilité de regrouper automatiquement des produits de même catégorie 
    - Réalisation d’une mesure pour confirmer ton analyse visuelle, en calculant la similarité entre les catégories réelles et les catégories issues d’une segmentation en clusters.

3) Suivant cette approche, démontrer la faisabilité de regrouper automatiquement des produits de même catégorie

4) Extraire les features texte :
    - Type “bag-of-words”, comptage simple de mots et Tf-idf ;
    - Une approche de type word/sentence embedding classique avec Word2Vec (ou Glove ou FastText) ;
    - Une approche de type word/sentence embedding avec BERT ;
    - Une approche de type word/sentence embedding avec USE (Universal Sentence Encoder).

5) Extraire les features image, il sera nécessaire de mettre en œuvre :
    - Un algorithme de type SIFT 
    - Un algorithme de type CNN Transfer Learning.

# Le jeu de données

Le jeu de données se compose de 1 fichier csv. C’est un jeu de données de 1050 articles comportant divers informations à leurs sujets tels que le nom du produit, une description, le lien vers l’articles etc…
Ce jeu de données est accompagné d’un dossier de 1050 images, correspondant chacune à un article du fichier csv.

# Compétences évaluées

- Évaluer les performances des modèles d’apprentissage non supervisé
- Sélectionner et entraîner des modèles d’apprentissage non-supervisé

# Livrables

1) Un notebooks contenant les fonctions permettant le prétraitement et la feature extraction des données textes
2) Un notebooks contenant les fonctions permettant le prétraitement et la feature extraction des données textes
3) Un notebook de classification supervisée des images
4) Un script Python (notebook et fichier .py) de test de l’API
5) Un support de présentation
