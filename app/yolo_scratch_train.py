

from IPython import display
display.clear_output()
import mlflow
from mlflow_utils import *
from mlflow.tracking import MlflowClient
import ultralytics
from ultralytics import settings
import yaml

# Update a setting
settings.update({"mlflow": False})
# settings.reset()
ultralytics.checks()



# from roboflow import Roboflow
# rf = Roboflow(api_key="VKgZy4Zly9FYuMlE2pkY")
# project = rf.workspace("myoptisol-oe4tj").project("mixed-merged")
# version = project.version(10)
# dataset = version.download("yolov8")

from ultralytics import YOLO
import os
HOME = os.getcwd()
print(HOME)

import yaml

# Load the YAML file
with open('args.yaml', 'r') as file:
    args_dict = yaml.safe_load(file)



artifact_path = "trolley-detection-pretrained-scratch-artifact"
experiment_id = create_mlflow_experiment("yolov8-scratch-trolley", artifact_path, {"yolov8": "trolley-detection-pretrained-scratch","dataset":"mixed-merged-10","version":"1"})

client = MlflowClient()
# mlflow.pytorch.autolog(registered_model_name="trolley-detection-pretrained-scratch")
# Load the model
model = YOLO('./model.yaml')

class YOLOv8Wrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load both YOLOv8 models
        self.best_model = YOLO(context.artifacts["best_model"])
        self.last_model = YOLO(context.artifacts["last_model"])

    def predict(self, context, model_input, model_type='best'):
        # Choose which model to use for prediction
        if model_type == 'best':
            results = self.best_model(model_input)
        elif model_type == 'last':
            results = self.last_model(model_input)
        else:
            raise ValueError("Invalid model_type. Choose 'best' or 'last'.")

        # Return predictions in a pandas DataFrame format
        return results.pandas().xyxy[0]

with mlflow.start_run(experiment_id=experiment_id) as run:
    results =  model.train(
        data=f'{HOME}/datasets/mixed-merged-10/data.yaml',  # Dataset path
        epochs=args_dict['epochs'],  # Number of epochs
        patience=100,  # Early stopping patience
        batch=-1,  # Automatic batch size
        imgsz=640,  # Image size
        save_period=5,  # Save checkpoint every 5 epochs
        cache='ram',  # Cache dataset in RAM
        device=0,  # GPU device ID
        workers=8,  # Number of data loading workers
        project=f'{HOME}/trolley-detection-pretrained-scratch',  # Project folder
        exist_ok=False,  # Don't overwrite existing project folder
        pretrained='yolov8n.pt',  # Pretrained weights path
        optimizer='auto',  # Optimizer type
        verbose=True,  # Verbose output
        seed=33,  # Random seed for reproducibility
        deterministic=True,  # Ensure deterministic training
        mask_ratio=4,  # Mask ratio for loss calculation
        dropout=0.0,  # Dropout rate
        val=True,  # Enable validation
        split='val',  # Validation split
        lr0=0.001,  # Initial learning rate
        lrf=0.01,  # Final learning rate
        momentum=0.95,  # Momentum for SGD
        weight_decay=0.01,  # Weight decay for regularization
        warmup_epochs=5,  # Warmup epochs
        warmup_momentum=0.8,  # Warmup momentum
        warmup_bias_lr=0.1,  # Warmup bias learning rate
        box=0.05,  # Box loss gain
        cls=0.5,  # Class loss gain
        dfl=1.5,  # Distribution Focal Loss (DFL) gain
        kobj=1.0,  # Keypoint objectness loss gain
        label_smoothing=0.1,  # Label smoothing factor
        tracker='botsort.yaml'  # Tracker config file path
    )
    # Log metrics, parameters, or artifacts as needed
    artifacts = {
    "best_model": f"{results.save_dir}/weights/best.pt",
    "last_model": f"{results.save_dir}/weights/best.pt"
}
    mlflow.log_params(args_dict)
    
    mlflow.log_artifacts(f"{results.save_dir}",artifact_path=artifact_path)
    # mlflow.log_artifact(f"{results.save_dir}/weights/best.pt",artifact_path=artifact_path+"/weights_best")
    
    # mlflow.log_artifact(f"{results.save_dir}/weights/last.pt",artifact_path=artifact_path+"/weights_best")
    # Don't forget to log your model at the end of training
    mlflow.pyfunc.log_model(artifact_path,python_model = YOLOv8Wrapper(),artifacts=artifacts,registered_model_name="trolley-detection-pretrained-scratch")
    # mlflow.pyfunc.log_model(artifact_path+"/weights",python_model = YOLOv8Wrapper(),artifacts=artifacts,registered_model_name="trolley-detection-pretrained-scratch")
    model_info = client.get_latest_versions("trolley-detection-pretrained-scratch")[0]
    ## UPDATE MODEL VERSION TAGS FOR THE LATEST MODEL
    client.set_model_version_tag("trolley-detection-pretrained-scratch",version= model_info.version,key= "models",value="best.pt, last.pt")
    # model.load(f"{results.save_dir}/weights/best.pt")
    # mlflow.pyfunc.log_model(artifact_path+"/weights",python_model = YOLOv8Wrapper(),artifacts=artifacts,registered_model_name="trolley-detection-pretrained-scratch")
    # client.set_model_version_tag("trolley-detection-pretrained-scratch",2,"version","best.pt")
    