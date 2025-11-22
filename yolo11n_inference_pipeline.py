from ultralytics import YOLO

# Sélectionne la meilleure version du modèle YOLO11n en fonction de son score F1
# Si dans Google Collab : /content/runs/detect/yolo11n_person_project/weights/best.pt
model = YOLO("/runs/detect/yolo11n_person_project/weights/best.pt")

# Récupère le path de l'image ou vidéo de test
# Si dans Google Collab : /content/person_detection_project/test_image_2.jpg
test_path = "/test_image_2.jpg"

# Affiche les résultats du nombre de personnes détectées
results = model.predict(conf = 0.05, source = test_path, save = True)
r = results[0]
classes = r.boxes.cls.tolist()
num_persons = classes.count(4)

print("Number of persons detected in the image :", num_persons)

# Message de fin
print("Inference done with the inference pipeline. Check the 'runs/predict' folder.")