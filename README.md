# House Price Prediction Pipeline 🏠 📈
An end-to-end Machine Learning system that predicts housing prices using a modular, reproducible pipeline. This project demonstrates intermediate Python proficiency, MLOps integration with DVC, and containerization with Docker.


# 🏗️ System Architecture (DAG)
The project follows a Directed Acyclic Graph (DAG) managed by DVC to ensure data reproducibility:

1. Splitting: Divides raw data into training and test sets.

2. Feature Engineering: Processes numerical and categorical data for the model.

3. Build Model: Trains a regression model using Scikit-learn.

4. Evaluate Model: Generates performance metrics to track model quality.

#🛠️ Tech Stack
Language: Python 🐍 (Pandas, Scikit-learn, Flask)

MLOps: DVC (Data Version Control)

DevOps: Docker 🐳

Frontend: HTML/CSS/JavaScript


# 🚀 How to Run with Docker
1. You can pull and run the pre-built image directly to see the web interface:

Pull the image:
```title-'Bash'

docker pull truthmyson/house_price_prediction:latest
```
2. Run the container:

```title='Bash'

docker run -p 2662:2662 truthmyson/house_price_prediction:latest
```
3. Access the App: Open your browser and navigate to http://localhost:2662
