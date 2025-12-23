from ultralytics import YOLO

def train_ambulance_model():
    # 1. Load the base model (pre-trained on COCO)
    # 'yolov8n.pt' is the nano version (fastest, good for real-time)
    model = YOLO("yolov8n.pt")  

    # 2. Train the model
    # You need a dataset.yaml file that points to your images.
    # DATASET GUIDE:
    # 1. Go to Roboflow Universe (https://universe.roboflow.com/)
    # 2. Search for "Ambulance" or "Emergency Vehicle"
    # 3. Download a dataset in "YOLOv8" format.
    # 4. Unzip it. It will have a 'data.yaml' file.
    # 5. Change 'path_to_data.yaml' below to point to that file.
    
    print("Starting Training...")
    results = model.train(
        data="path/to/your/dataset/data.yaml", # <--- UPDATE THIS PATH
        epochs=50,          # 50 epochs is usually good for a draft model
        imgsz=640,          # Standard image size
        device='cpu',       # Use '0' for GPU if you have NVIDIA, else 'cpu'
        project="traffic_project",
        name="ambulance_model"
    )
    
    print("Training Complete!")
    print(f"Your new model is saved at: traffic_project/ambulance_model/weights/best.pt")
    print("Rename 'best.pt' to 'ambulance.pt' and place it in the same folder as main.py")

if __name__ == "__main__":
    train_ambulance_model()
