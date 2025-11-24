import os
import kagglehub
from ultralytics import YOLO

# Télécharge la dernière version du dataset sur Kaggle dans un dossier cache si ce n'est pas déjà fait
dataset_path = kagglehub.dataset_download("yusufberksardoan/traffic-detection-project")
print("Path to dataset files:", dataset_path)
path_data_yaml = os.path.join(dataset_path, "data.yaml")

# Sélectionne le modèle YOLO11n
model = YOLO("yolo11n.pt")
model.to("cuda") # pour forcer à utiliser le GPU T4 sur Google Collab

# Entraîne le modèle avec le dataset sur Kaggle
total_epochs = 50

model.train(
    data = path_data_yaml,
    epochs = total_epochs,
    imgsz = 640,
    batch = 16,
    name = "yolo11n_person_project",
    device = 0, # Utilisant le GPU
)

# Retour sur le score F1 après l'entraînement
metrics = model.val()
f1_person = metrics.box.f1[0]
print(f"\n\nF1 (person) after {total_epochs} epochs : {f1_person:.4f}")