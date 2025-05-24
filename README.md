# Bosch-price-optimizer
Pricing Optimization to simulate Bosch Corporation price prediction and optimization.
A Flask-based web service that predicts optimal product prices using a machine learning model trained on Bosch sales data (using a randomly extracted csv file from amazon sales data downloaded from kaggle).

## Live Demo

> Deployed on [Render](https://render.com/) 

## Features

- REST API for price prediction
- XGBoost model loaded via `joblib`
- Hosted with Flask and Gunicorn

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
