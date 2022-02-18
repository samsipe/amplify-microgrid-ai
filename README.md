# Amplify Microgrid AI
- [Amplify Microgrid AI](#amplify-microgrid-ai)
  - [Overview](#overview)
    - [Project Details](#project-details)
  - [Setup and Installation](#setup-and-installation)
  - [Training](#training)
  - [Jupyter Notebooks](#jupyter-notebooks)
  - [Deployment](#deployment)
    - [Dash App - Docker Compose](#dash-app---docker-compose)
    - [Dash App - Kubernetes](#dash-app---kubernetes)

## Overview
**This project aims to modernize how buildings interact with the grid**, utilizing deep learning techniques to predict future usage and generation to make smarter energy decisions.

### Project Details
[Google Drive](https://drive.google.com/drive/folders/1sVjw4bLe3xxM489szpL0qAIXmHHzE7Xp?usp=sharing)

## Setup and Installation
Run the following command to get `amplify` installed in your environment.

```shell
virtualenv -p python3.8 .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
Now setup your ClearML integration, go to your [ClearML Dashboard](https://app.clear.ml/dashboard) and run:
``` shell
clearml-init
```
From there you will need to add AWS Access Keys to access models stored in S3.

## Training
Training can be completed using ClearML tasks or inside Jupyter Notebooks

## Jupyter Notebooks
These are found [here](notebooks/) and  used for development, model training, and testing. All functionality should be migrated to the [amplify](amplify/) module.

## Deployment

This model has been packaged in a Dash application that can be deployed in a container.

### Dash App - Docker Compose
To deploy with Docker Compose, first make a file called `.env` and input keys/secrets for [OpenWeather](https://openweathermap.org/api/one-call-api), ClearML, and AWS.

```
OW_API_KEY=
CLEARML_WEB_HOST=https://app.clear.ml
CLEARML_API_HOST=https://api.clear.ml
CLEARML_FILES_HOST=https://files.clear.ml
CLEARML_API_ACCESS_KEY=
CLEARML_API_SECRET_KEY=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=us-east-1

```
Then run the following:

```
docker-compose up -d --build
```

### Dash App - Kubernetes

First Create a cluster

```
eksctl create cluster -f eks_cluster.yml --profile xyz
```
