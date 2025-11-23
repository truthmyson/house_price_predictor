# End-to-End House Price Prediction Project
## 🏠 Overview
This project is a complete, end-to-end Machine Learning solution for predicting house prices. It encapsulates the entire data science lifecycle, from initial data ingestion and analysis to model deployment and serving predictions through a web interface. Built with software engineering best practices, it features a modular Object-Oriented Programming (OOP) design, experiment tracking with MLflow and DagsHub, and containerization with Docker.

The application provides a user-friendly frontend where users can input house features and receive an instant price prediction, powered by a Flask API and a trained Scikit-learn regression model

## ✨ Features
Data Flow Management: Custom OOP classes for robust data ingestion and processing.

Exploratory Data Analysis (EDA): In-depth analysis and visualization of the dataset using Pandas and Matplotlib.

Machine Learning Modeling: A regression model built with Scikit-learn to predict house prices.

MLOps & Experiment Tracking: Integrated with MLflow and DagsHub to log experiments, parameters, metrics, and models for full reproducibility and comparison.

Model Serving: A RESTful API built with Flask to serve model predictions.

Interactive Frontend: A clean and responsive web interface built with HTML, CSS, and JavaScript for user interaction.

Containerization: The entire application is dockerized using Docker, ensuring consistent execution across any environment.

## 🗂️ Project Architecture & Workflow
The project follows a logical, modular pipeline:



<img width="3977" height="917" alt="deepseek_mermaid_20251123_40d7b4" src="https://github.com/user-attachments/assets/f48af5ae-ea11-4d68-8b70-eb8c653dfec0" />






Data Management: OOP classes handle the loading and validation of the dataset.

Analysis & Training: The data is explored, a model is trained, and all artifacts are logged to DagsHub via MLflow.

Deployment: The trained model is packaged with a Flask app to create a prediction endpoint.

Interaction: The frontend sends user input to the Flask endpoint and displays the prediction.

Containerization: All components are bundled into a single Docker image for easy deployment.

## 🛠️ Technology Stack
Category	|  Technologies.

Machine Learning	|  Python, Scikit-learn, Pandas, NumPy, Matplotlib.

MLOps & Tracking	|  MLflow, DagsHub.

Backend & API	 |  Flask.

Frontend	|  HTML5, CSS3, JavaScript.

Containerization	|  Docker.

Development	 |  Object-Oriented Programming (OOP).


## 🚀 Getting Started
### Prerequisites
Docker installed on your machine. <a href="https://docs.docker.com/get-started/get-docker/">Get Docker</a>

### Installation & Run
1. Clone the repository

  git clone [https://github.com/your-username/your-repo-name.git](https://github.com/truthmyson/house_price_predictor)

cd your-repo-name

2. Build the Docker image

3. Run the container

4. Open your browser and navigate to http://localhost:5000 to view the application and start making predictions!
