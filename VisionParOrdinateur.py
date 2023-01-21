import cv2
import numpy as np
import tensorflow as tf 
from collections import Counter

tab = np.linspace(0,10,11)
tab_val = []

# Charger le modèle de reconnaissance de chiffres pré-entraîné
model = tf.keras.models.load_model("C:\\Users\\cano9\\Jupyter lab\\Projet ML\\mon_modele_mnist_convolutifs3.h5")

# Ouvrir la caméra ordinateur
cap = cv2.VideoCapture(0)

# Vérifier si la vidéo a été ouverte correctement
if not cap.isOpened():
    print("Erreur lors de l'ouverture de la vidéo")
    exit()

i = 0
while True:
    # Lire une image depuis la caméra
    ret, frame = cap.read()
    
    # Prétraiter l'image en niveaux de gris et en ajustant sa taille
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (28, 28))
    
    # Afficher le frame
    cv2.imshow("Frame", frame)
    
    # Prédire le chiffre présent sur l'image
    input_img = np.expand_dims(gray, axis=0)
    prediction = tab[np.argmax(model.predict(input_img))]
    print("Prédiction de cette image :", prediction)
    tab_val.append(prediction)

    
    # Quitter si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Relâcher la caméra et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()

# Utilisez la fonction Counter pour compter les occurrences de chaque valeur dans le tableau
counts = Counter(tab_val)

# Trouvez la longueur totale du tableau
length = len(tab_val)

# Pour chaque valeur dans le tableau, calculez et affichez sa proportion dans le tableau
for value, count in counts.items():
    proportion = count / length
    print(f"La valeur {value} représente {proportion:.2f} du tableau")