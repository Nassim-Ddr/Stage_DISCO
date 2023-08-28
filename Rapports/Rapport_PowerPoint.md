# Logiciel de présentation: Powerpoint

( #### **Image de PowerPoint** ######)

# Commandes implémentés
Pour un logiciel de présentation qui serait semblable à PowerPoint, on choisit de nous concentrés sur un nombre limité de commandes. Pour des raisons de temps et de praticité. Voici donc les commandes qui nous ont parue le plus pertinant:
- Alignement des objets ( transformations spatiales ): permet d'aligner parfaitement les objets. Au lieu de faire bouger manuellement les objets pour les aligner. Les alignements possibles sont:
    - aligner en haut / en bas
    - aligner à gauche / à droite
- Copier & Aligner (CTRL + Drag): Permet de dupliquer puis faire bouger avec la souris, en gardant d'alignement  avec les objets copiés. Par exemple lorsqu'on veut faire une grille ( ###### **EXEMPLE IMAGE** ########)
- Premier Plan / Arrière-Plan: Permet de changer la profondeur des objets.  
Scénario simple: Trois objets A,B,C l'une sur l'autre, l'utilisateur met B et C au premier - plan.  On peut faire la même résultat en mettant A en arrière plan. ( ###### **EXEMPLE IMAGE Avant-Après** ########)

# Partie Modèle
## 2 approches pour la prédiction
Pour l'implémentation du modèle à utiliser 2 approches différentes pour prédire les commandes: l'un avec du machine learning ( réseau de convolutions ), et l'autre avec un algorithme codé à la main.

<!---
Machine Learning:
- Avantages: 
    - plus facile à généraliser sur plus de commandes. On doit juste avoir l'effet de la commande pour l'apprentissage
- Désavantages: 
    - besoin d'un dataset
    - temps de prédiction qui peut être long ( dépend de l'architecture du modèle )
    - peut faire erreurs de prédictions et dur à debugger

Hardcode:
- Avantages:
    - facile à débugger
    - dans le cas de PowerPoint, avec les commandes qu'on à choisit. On peut assez aisément le faire avec des conditions (if et else)
    - pas besoin de dataset
- Désavantages:
    - Moins généralisable. Besoin de coder les règles pour chaque commande possibles
    - besoin d'avoir accès au objets du logiciel de présentation
--->
| Machine Learning                                   | Hardcode                                      |
| -------------------------------------------------- | --------------------------------------------- |
| **Avantages:**                                    | **Avantages:**                               |
| - plus facile à généraliser sur plus de commandes. On doit juste avoir l'effet de la commande pour l'apprentissage | - facile à débugger                           |
| **Désavantages:**                                 | **Désavantages:**                            |
| - besoin d'un dataset                            | - Moins généralisable. Besoin de coder les règles pour chaque commande possibles |
| - temps de prédiction qui peut être long (dépend de l'architecture du modèle) | - besoin d'avoir accès aux objets du logiciel de présentation |
| - peut faire des erreurs de prédictions et difficile à debugger | - dans le cas de PowerPoint, avec les commandes qu'on a choisies. On peut assez aisément le faire avec des conditions (if et else) |

## Etat de l'application
<!-- Machine Learning
- état: image du slide de présentation
- pourquoi: 
   - contraitement au hardcode, on ne peut pas prendre une liste avec un nombre d'objet qui change, sauf avec les transformers et RNN
   - image est un état fini, bon input pour un modèle de machine learning


Hardcode
- état: liste des formes avec leurs propriétés (couleurs, dimensions, position, ...)
- pourquoi: parceque c'est comme ça -->
|            | Machine Learning                | Hardcode                                 |
|------------| ------------------------------- | ---------------------------------------- |
| **État:**  | <li>Image du slide de présentation | <li> Liste des objets avec leurs propriétés (forme, couleurs, dimensions, position, ...) |
| **Pourquoi:**  |  <li>Contrairement au hardcode, on ne peut pas prendre une liste avec un nombre d'objets qui change, sauf avec les RNNs et transformers <li> l'image est un état fini, bon input pour un modèle de machine learning | <li> parce que c'est comme ça     |


## Modele
### Machine Learning
#### Generation des données
( ####### A mieux expliquer #######)  
Pour notre modèle de machine learning, nous aurons besoin d'une base de données dont laquelle il peut apprendre dessus. Donc la base de données contiendra une image avant et après le lancement des commandes qu'on a choisis. Nous créons nous même les données en plaçant des formes aléatoires avec des positions et tailles aléatoires, qu'on lance ensuite un des commandes choisis. Pendant la génération des objets, on empêche la création d'objet contenu ou contenant un autre objet. Afin de pouvoir apprendre les commandes d'alignement. 
Pour l'apprentidsage de notre modèle, nous avons générés (### Nombre de dataset ####) images.

Hardcode: pas besoin **obviously**

#### Traitement des données
- Normalisation
- Cropping
- Take Moving Only

#### Architecture du modèle
Machine Learning: réseaux de convolution (CNN) :
1. Input Layer:
   - Input shape: [Batch Size, 64, 64, 3] for RGB images
   
2. Convolutional Layers:
   - Convolutional layer with 6 filters, kernel size (5x5), stride 1, padding 2
   - Activation function: ReLU
   - MaxPooling layer with 2x2 pool size
   
3. Convolutional Layers:
   - Convolutional layer with 16 filters, kernel size (5x5), stride 1
   - Activation function: ReLU
   - MaxPooling layer with 2x2 pool size
   
4. Fully Connected Layers:
   - Flatten the output from the previous layer
   - Dense (fully connected) layer with 1024 units
   - Activation function: ReLU

5. Fully Connected Layers:
   - Dense (fully connected) layer with 256 units
   - Activation function: ReLU

6. Output Layer:
   - Dense (fully connected) layer with 4 units
   - Activation function: Softmax (for multiclass classification)

6. Model Compilation:
   - Loss function: Categorical Cross-Entropy
   - Optimizer: Adam
   
On s'est inspiré de LeNet pour créer ce modèle  
(#### **Image LeNet ou je sais** ####)

### Hardcode
- Input: liste des objets avec leurs propriétés: ( avant et après )
   - type de objets
   - position du rectangle englobant
   - couleurs de remplissage RGB
   - couleur du contour RGB


(### On prend seulement le premier objet du résultat)
- Prédiction alignement:
   1. obtenir les objets qui ont changés, en comparant avec les objets d'avant.
   2. L'objet qui a changé, __o__ regarde si il y a alignement possible (paramètre __eps__) avec un autre objets présents dans la liste

- Prédiction Copy-Align:
   1. Obtenir l'objet qu'on focus:
      - Si la taille avant et après est la même, on compare avec les objets d'avant. On devrait pouvoir obtenir seulement l'objet qui a changé
      - Si la taille de la liste a seulement augmenter de 1, on récupère l'objet récemment crée sur le slide
   
   2. L'objet qu'on a choisit de focus, __o__, on regarde si il y a objet similaire (paramètre __eps__) présents dans la liste et qui est alignés avec un autre objet

- Prédiction Foreground-Background
   1. Si déplacement des objets ou changement dans les propritétés (couleurs, ...), alors retourne Faux
   2. calculer la matrice des relations avant et après. Matrice des relations: Soit f(i,j) = 
      - 1 si o_i devant o_j
      - -1 si o_i derrière o_j
      - 0 si o_i = o_j ou o_i et o_j ne s'intersecte pas ( = pas de relation)
   3. Prendre que les objets qui ont changé de relations
   4. Prendre celui avec le plus de changement
 
## Evalutation modèle
### ML
- accuracy, ...
- temps de  prédiction

Evaluation sur les données de test
1. Model Alignement avec état courant seulement
Test performance: 0.775
========= Precision ==============
alignBottom : 0.815
alignLeft : 0.811
alignRight : 0.734
alignTop : 0.743

2. Model Alignement avec état courant + état précedent (selection tout)
Test performance: 0.939
========= Precision ==============
alignBottom : 0.979
alignLeft : 0.938
alignRight : 0.954
alignTop : 0.887

3. Model Alignement avec état courant + état précedent (selection partielle)
Test performance: 0.895
========= Precision ==============
alignBottom : 0.915
alignLeft : 0.896
alignRight : 0.889
alignTop : 0.878

Evaluation en pratique
1. détecte bien les alignement en pratique, si l'alignement se fait sur tous les objets présents. A des difficultés pour les alignements sur seulement une partie des objets du diapositives

2. détecte les alignements, mais seulement lorsque on fait un déplacement semblable aux commandes d'alignements (exemple déplacer les objets en diagonales peut causer des problèmes car le classifier n'a jamais eu ces types de données)

3. détecte les alignements et meilleure que le modèle 2. lorsqu'on aligne une partie des objets 

=> model 2 et 3 prend en compte le déplacement des objets, il semblerait comme on l'entraîne sur la commande alignement, il a prends plus en compte le déplacement des objets plutôt que  l'état final ce qui est correcte mais faux également. Par exemple, si on fait un alignement à droite à la main en déplaçant l'objet vers la gauche, le modèle a plus tendance à prédire que c'est un alignement à gauche. Cela s'expplique par la commande d'alignement, car un alignement à droite, s'accompagne toujours d'un déplacement d'un objet vers la droite des objets leplus à gauche plutôt que des objets le plus à droite.


### Hardcode
- Points où ça marche
- Points où ça marche pas

# Recommender
- temps de réponse
## comment ça fonctionne ?

# Conclusion
- ouverture, points à améliorer, ce qu'on peut faire
- difficultés
- ce que t'as appris
- conclusion






# To Do
Ecrire l'impact de la fonctionnalité et le temps