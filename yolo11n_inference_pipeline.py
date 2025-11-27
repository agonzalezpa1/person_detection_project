import os
from ultralytics import YOLO

# Sélectionne la meilleure version du modèle YOLO11n en fonction de son score F1
model = YOLO("/runs/detect/yolo11n_person_project/weights/best.pt")
# Si dans Google Collab : /content/runs/detect/yolo11n_person_project/weights/best.pt

# Récupère le path de l'image de test (ici 'test_image_2.jpg')
test_path = "/test_images/test_image_2.jpg"
# Si dans Google Collab : /content/person_detection_project/test_images/[nom_image].jpg
file_name = os.path.splitext(os.path.basename(test_path))[0] # Récupère le nom de l'image

# Affiche les résultats du nombre de personnes détectées et sauvegarde l'image annotée
results = model.predict(conf = 0.05, source = test_path, save = True)
r = results[0]
classes = r.boxes.cls.tolist()
num_persons = classes.count(4) # Nombre de personnes détectées

print(f"\n\nNumber of persons detected in the image '{file_name}' :", num_persons)