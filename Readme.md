### YOLOv8 Model Training, Tracking, and Deployment

This repository contains everything you need to train, track, and deploy a YOLOv8 model using MLflow. This setup ensures that your machine learning experiments are reproducible and easy to manage, while providing a streamlined process for deploying your models.

## Getting Started

### Prerequisites

Ensure you have the following installed on your system:

- **Docker**: For containerizing the environment and ensuring reproducibility.
- **MLflow**: For tracking experiments, storing artifacts, and managing deployments.
- **Python 3.8+**: Required for running the YOLOv8 model training.

### Repository Structure

- **Dockerfile**: Defines the Docker image that will be used for the training environment.
- **MLproject**: Configuration file for MLflow that specifies the entry points, dependencies, and environment setup.
- **yolo_scratch_train.py**: Script to train the YOLOv8 model from scratch, utilizing the configurations specified in MLproject.
- **datasets/**: Directory where your training datasets should be placed or mounted.
- **inference.py**: Script to request to the `/invocations` endpoint.

### Step-by-Step Guide

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd yolov8-training
```

#### 2. Start the MLflow Tracking Server

MLflow needs a backend to store the experiment results and artifacts. Start the MLflow server with the following command:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 5000 &
```

This command will start the MLflow server on `http://0.0.0.0:5000`, accessible from any IP address on your local network.

#### 3. Set the MLflow Tracking URI

Set the MLflow tracking URI to point to the server you've just started:

```bash
export MLFLOW_TRACKING_URI="http://<host-ip>:5000"
```

Replace `<host-ip>` with the actual IP address of your host machine.

#### 4. Build the Docker Image

Use the provided `Dockerfile.train` to build the Docker image:

```bash
docker build -f Dockerfile.train -t yolo_train_scratch:v1 .
```

This command creates a Docker image that contains all dependencies required for YOLOv8 training.

#### 5. Run the Training Experiment

Now you can run the YOLOv8 training experiment using MLflow. The MLflow run command will start a Docker container using the image you built:

```bash
mlflow run --experiment-name yolov8-scratch . -A gpus=all -A shm-size=8g
```

- **`--experiment-name yolov8-scratch`**: Specifies the experiment name under which results will be logged.
- **`-A gpus=all`**: Allocates all available GPUs to the Docker container for training.
- **`-A shm-size=8g`**: Increases the shared memory size to prevent memory-related errors during training.


### Deploying the YOLOv8 Model with Custom Docker Image

You can deploy the YOLOv8 model using a Docker container with GPU support. The Dockerfile is designed to accept a custom `MLFLOW_MODEL_URI` at runtime, making it flexible for deploying any model logged in MLflow.


#### 1. Build the Docker Image

Build the Docker image:

```bash
docker build -f Dockerfile.serve -t trolley-model-deploy .
```

#### 2. Run the Docker Container with a Specific Model URI

You can deploy the model by running the Docker container and passing the specific `MLFLOW_MODEL_URI` as an environment variable:

```bash
docker run --gpus all -p 5001:8080 -e MLFLOW_MODEL_URI="runs:/<your-run-id>/<artifact-name>/weights"  -e MLFLOW_TRACKING_URI=http://<host-ip>:5000 trolley-model-deploy
```

- **`--gpus all`**: Allocates all available GPUs to the Docker container.
- **`-e MLFLOW_MODEL_URI="runs:/<your-run-id>/<artifact-name>/weights"`**: Sets the `MLFLOW_MODEL_URI` for the model you want to serve. Replace `<your-run-id>` with the actual run ID from MLflow,<artifact-name> to your artifact name,
and `<host-ip>` with the actual IP address of your host machine.

#### 4. Access the Deployed Model

To access the deployed YOLOv8 model, you can send a POST request to the `/invocations` endpoint. Below is a dummy example of how you would structure such a request considering the `YOLOv8Wrapper` model class:

```python
import requests
import json

# Example input data format for YOLOv8
input_data = {
    "columns": ["image_data"],
    "data": [["base64_encoded_image_string_or_image_path"]],
    "model_type": "best"  # Specify which model to use: 'best' or 'last'
}

# URL of the deployed model
url = "http://localhost:5001/invocations"

# Make the POST request to the deployed model
response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(input_data))

# Parse the response
predictions = response.json()
print(predictions)
```

This dummy request assumes the following:

- **Input Data**: The request contains a base64-encoded image string or a path to the image, depending on how your model expects to receive image data.
- **Model Type**: You can specify which YOLOv8 model to use (`'best'` or `'last'`).


### Troubleshooting

- **Connection Issues**: Ensure that the IP addresses and ports are correctly configured, especially when running Docker containers and accessing MLflow.
- **Memory Errors**: Adjust the `shm-size` parameter if you encounter shared memory issues during training.

### Conclusion

This setup allows you to efficiently manage and track your YOLOv8 training experiments with MLflow, ensuring that your results are reproducible and easily deployable. By containerizing the environment with Docker, you can ensure consistency across different systems and streamline the deployment process.

---

Feel free to customize the steps according to your specific needs, such as modifying Docker images or adjusting MLflow configurations. Happy training!

