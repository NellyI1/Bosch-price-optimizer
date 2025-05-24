# Bosch-price-optimizer
Pricing Optimization to simulate Bosch Corporation price prediction and optimization.
A Flask-based web service that predicts optimal product prices using a machine learning model trained on Bosch sales data (using a randomly extracted csv file from amazon sales data downloaded from kaggle).

## Live Demo

> Deployed on [Render](https://render.com/), a cloud platform for hosting web applications.

## Features

- REST API endpoint for price prediction  
- Uses a pre-trained XGBoost model loaded via `joblib`  
- Flask web server with Gunicorn for production-ready hosting  
- Easily extendable for new pricing models or datasets  

## Installation and Setup

### Prerequisites

- Python 3.8 or higher  
- pip package manager  

### Clone the repository

```bash
git clone https://github.com/NellyI1/Bosch-price-optimizer.git
cd Bosch-price-optimizer

