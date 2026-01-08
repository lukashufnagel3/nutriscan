# NutriScan: AI-Powered Food Recognition & Nutritional Estimation

NutriScan is a Proof-of-Concept (PoC) application designed to recognize food items from images and estimate their macronutrients, featuring a fully containerized deployment of a Vision Transformer (ViT) model fine-tuned on the Nutrition14 dataset.

The application features a user-friendly web interface built with Streamlit and is packaged with Docker for easy portability and deployment.
## Project Structure

Ensure your project directory is organized as follows before building:

NutriScan_Deploy/  
├── model/  
│   ├── config.json   
│   ├── preprocessor_config.json  
│   └── model.safetensors  
├── app.py                    
├── Dockerfile                
├── nutrition_data.json       
├── requirements.txt          
├── requirements.graphic  
└── README.md                 

## Prerequisites

- Docker: Ensure Docker is installed and running on your machine.
- Git LFS: Make sure Git Large File Storage is installed and initialized.


## Quick Start (Docker)

Follow these steps to build and run the application.
### 1. Build the Docker Image

Navigate to the project folder and run the build command. This creates an image named nutriscan-app.


`docker build -t nutriscan-app .`

> Note:
> The first build may take a few minutes as it downloads the base Python image and the PyTorch libraries. Subsequent builds will be much faster due to caching.

### 2. Run the Container

Start the application by mapping port 8501 (Streamlit's default) from the container to your local machine.

`docker run -p 8501:8501 nutriscan-app`

### 3. Access the App

Once the container is running, open your web browser and navigate to:

http://localhost:8501
## How to Use
### 1. Upload an Image
Drag and drop a food image (JPG/PNG) into the upload area, or browse from your files.  
### 2. View Preview
The app will display the image to confirm it loaded correctly.  
### 3. Analyze
Click the "Analyze Nutrition" button.  
### 4. View Results:
The app will display the following:  
- Identified Class: The predicted food category (e.g., RICE, COOKING VEGS).  
- Macronutrients: Estimated Calories, Protein, Carbs, and Fat per 100g.  
- Macro Ratio: A donut chart visualizing the macronutrient distribution.  

## Technical Details

### Model Architecture: 
Vision Transformer (google/vit-base-patch16-224) fine-tuned on the Nutrition14 dataset.
### Deployment: 
The Docker image is optimized for CPU inference to ensure compatibility across different hardware without requiring NVIDIA drivers.
### Optimization: 
The build process uses a specific layering strategy to cache dependencies, allowing for rapid updates to the app.py code without re-downloading heavy libraries like PyTorch.

### Troubleshooting "Unknown Labels"

If the application predicts an "Unknown" class (e.g., Class 12), it means the model's internal configuration does not contain the semantic names.

>Fix: The app.py includes a fallback manual mapping list (LABELS) derived from the training dataset. Ensure this list matches the alphabetical order of your classes.