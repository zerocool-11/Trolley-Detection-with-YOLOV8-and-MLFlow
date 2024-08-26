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
