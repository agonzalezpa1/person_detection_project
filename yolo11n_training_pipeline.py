import os
import kagglehub
from ultralytics import YOLO

# Télécharge la dernière version du dataset sur Kaggle dans un dossier cache si ce n'est pas déjà fait
dataset_path = kagglehub.dataset_download("yusufberksardoan/traffic-detection-project")
print("Path to dataset files:", dataset_path)
path_data_yaml = os.path.join(dataset_path, "data.yaml")

# Sélectionne le modèle YOLO11n
model = YOLO("yolo11n.pt")

# Entraîne le modèle avec le dataset sur Kaggle
total_epochs = 50

print(f"\n------ Epoch 1 /{total_epochs} -------------------------")

model.train(
    data = path_data_yaml,
    epochs = 1,
    imgsz = 640,
    batch = 16,
    name = "yolo11n_person_project",
    device = "cpu", # Utilisant le CPU de mon ordinateur
)

for epoch in range(2, total_epochs):
    print(f"\n------ Epoch {epoch}/{total_epochs} -------------------------")

    model.train(
        data = path_data_yaml,
        epochs = 1,
        imgsz = 640,
        batch = 16,
        name = "yolo11n_person_project",
        device = "cpu", # Utilisant le CPU de mon ordinateur
        resume = True # Pour continuer le training
    )

    # Retour sur le score F1 après chaque epoch
    metrics = model.val()
    f1_person = metrics.box.f1[0]
    print(f"F1 (person) after epoch {epoch + 1} : {f1_person:.4f}")

# Message de fin
print("Training completed using the training pipeline.")