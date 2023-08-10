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

#### Apprentissage
...

### Hardcode
- ...

## Evalutation modèle
### ML
- accuracy, ...
- temps de  prédiction

### Hardcode
- 

# Recommender
- temps de réponse
- comment ça fonctionne ?
- 


# Conclusion
- ouverture, points à améliorer, ce qu'on peut faire
- difficultés
- ce que t'as appris
- conclusion






# To Do
Ecrire l'impact de la fonctionnalité et le temps