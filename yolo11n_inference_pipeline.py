from ultralytics import YOLO

# Sélectionne la meilleure version du modèle YOLO11n en fonction de son score F1
model = YOLO("runs/detect/yolo11n_person_project/weights/best.pt")

# Récupère le path de l'image, dossier ou vidéo de test
source = "test.jpg"

# Affiche les résultats
results = model.predict(conf=0.25, source=source, save=True)
print(results)

# Message de fin
print("Inference done with the inference pipeline. Check the 'runs/predict' folder.")