import os
from ultralytics import YOLO

# Sélectionne la meilleure version du modèle YOLO11n en fonction de son score F1
# Si dans Google Collab : /content/runs/detect/yolo11n_person_project/weights/best.pt
model = YOLO("/runs/detect/yolo11n_person_project/weights/best.pt")

# Récupère le path de l'image de test
# Si dans Google Collab : /content/person_detection_project/[nom_image].jpg
test_path = "/test_image_2.jpg"
file_name = os.path.splitext(os.path.basename(test_path))[0]

# Affiche les résultats du nombre de personnes détectées
results = model.predict(conf = 0.05, source = test_path, save = True)
r = results[0]
classes = r.boxes.cls.tolist()
num_persons = classes.count(4)

print(f"\n\nNumber of persons detected in the image '{file_name}' :", num_persons)