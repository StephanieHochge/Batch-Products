# Batch Image Classification App

This application provides a system for automatic classification of returned products based on image data. It is designed to run as a batch process overnight and aims to reduce manual workforce in sorting and categorizing items. The application is built with Python 3.12 and uses a trained machine learning model (ONNX format) to classify images into predefined categories. The model is exposed via a RESTful API and supports both local and containerized deployments. 

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Running the App Locally](#running-the-app-locally)
3. [Running the App with Docker](#running-the-app-with-docker)
4. [Automation Using Jenkins](#automation-using-jenkins)
5. [Testing](#testing)
6. [License](#license)

---

## System Requirements

To run the app locally or with Docker/Jenkins, the following requirements must be met:

### **For Local Execution**
- Python 3.12 or higher
- Pip 
- Virtualenv (recommended for local installation)

### **For Docker Execution**
- Docker (minimum version 28.1.1)
- Docker Compose (minimum version 2.35.1)

### **For Automatic Execution**
- Jenkins (minimum version 2.492.1)

---

## Running the App Locally

1. **Clone the Repository:**
   ```sh
   git clone <repository-url>
   cd <your-project-folder>
   ```

2. **Create and Activate a Virtual Environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate  # Windows
   ```

3. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
   Note: For model training (not required for app execution), use `requirements-train.txt`. The installation of torch
   and torchvision depends on your operating system and whether you use a GPU (CUDA) or CPU. Please install them 
manually using the appropriate command from the official [PyTorch website](https://pytorch.org/get-started/locally/).
   

4. **Adjust Configuration:**

   Update the configuration files under `/configs` to ensure correct file storage and access.
   Edit the following two files to reflect the correct paths and access points on your local machine: 
     - `app_config.yaml`
     - `test_app_config.yaml`


5. **Create Test Data:**

   Create test data to simulate image ingestion and to run tests. 
   ```sh
   python -m tests.create_test_data
   ```


6. **Start the App:**
   ```sh
   python -m src.api.run
   ```
   The app should now be accessible at `http://localhost:5000`. You can view the database contents by clicking on `Show Database` or by visiting `http://localhost:5000/view_db`.


7. **Use the App for Ingestion and Batch Processing:**
    
    Manually start ingestion to simulate new image arrival:
    ```sh
    python -m src.triggers.ingest_trigger
    ```
   
    Start batch classification of ingested images:
    ```sh
    python -m src.triggers.batch_trigger
    ```

---

## Running the App with Docker

1. **Build the Docker Image:**
   ```sh
   docker build -t batch_app .
   ```

2. **Start the App and Initialize Test Data Using Docker Compose:**
   ```sh
   docker-compose up -d
   ```

3. **Verify that the Containers are Running:**
   ```sh
   docker ps
   ```

    The app should now be accessible at `http://localhost:4000`. You can view the database contents by clicking on `Show Database` or by visiting `http://localhost:4000/view_db`.


4. **Use the App for Ingestion and Batch Processing:**
    
    Manually start ingestion to simulate new image arrival:
    ```sh
    docker exec -it bp_app python -m src.triggers.ingest_trigger
    ```
   
    Start batch classification of ingested images:
    ```sh
    docker exec -it bp_app python -m src.triggers.batch_trigger
    ```
   
    Note: Docker manages four persistent volumes for storage:
    - Test data volume: Contains sample images.
    - App data volume:
      - `incoming_images/`: Raw received images.
      - `processed_images/`: Images classified by the model (in overnight batches).
      - `logs/`: The `app.log` file contains information about the app usage, while the `trigger.log` file provides details on the usage of the ingestion and batch processing triggers.
    - Model volume: Stores the ONNX model used for classification that was previously trained with the Fashion MNIST dataset. 
    - PostgreSQL volume: Used to persist PostgreSQL database data across container restarts. 
---

## Automation Using Jenkins

Three Jenkinsfiles are available in the `/jenkins` directory: 

1. **Image Ingestion Pipeline**
    - Triggers ingestion of new test images into `data/incoming_images`.
    - Scheduled twice daily at around 8:00 AM and 8:00 PM.

2. **Batch Processing Pipeline**
    - Executes image classification using the model.
    - Moves processed images to the `data/processed_images` folder.
    - Scheduled daily at 2:00 AM. 

3. **Testing Pipeline**
    - Executes unit tests to ensure system stability after updates.
    - Can be triggered manually. 

Jenkins must be installed and properly configured. Refer to [Jenkins Installation Guide](https://www.jenkins.io/doc/book/installing/). 
After installation, use the three Jenkinsfiles to create three Jobs via Jenkins GUI.

---
## Testing

Unit tests are implemented with `pytest`.
- local execution: 
```sh
pytest
```
- Docker execution:
```sh
docker exec -it bp_app pytest
```

---
## License

This application is part of a data science project and is not licensed for public or commercial use. 
