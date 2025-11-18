import os
import kagglehub
from ultralytics import YOLO

# Télécharge la dernière version du dataset sur Kaggle dans un dossier cache si ce n'est pas déjà fait
dataset_path = kagglehub.dataset_download("yusufberksardoan/traffic-detection-project")
print("Path to dataset files:", dataset_path)
path_data_yaml = os.path.join(dataset_path, "data.yaml")

# Sélectionne le modèle YOLO11n
model = YOLO("yolo11n.pt")

# Entraîne le modèle avec un retour sur le score F1 après chaque epoch
def log_f1(epoch) :
    metrics = model.val() # Récupère les métriques du modèle
    f1_person = metrics.box.f1[0]
    print(f"Epoch {epoch + 1} : F1 (person) = {f1_person:.4f}")

# Entraîne le modèle avec le dataset sur Kaggle
model.train(
    data = path_data_yaml,
    epochs = 50,
    imgsz = 640,
    batch = 16,
    name = "yolo11n_person_project",
    device = 0,
    callcallbacks = [log_f1]
)

# Message de fin
print("Training completed using the training pipeline.")