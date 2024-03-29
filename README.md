# Amplify Microgrid AI

- [Amplify Microgrid AI](#amplify-microgrid-ai)
  - [Overview](#overview)
    - [Project Details](#project-details)
  - [Setup and Installation](#setup-and-installation)
  - [Training](#training)
  - [Jupyter Notebooks](#jupyter-notebooks)
  - [Deployment](#deployment)
    - [Dash App - Docker Compose](#dash-app---docker-compose)
    - [Credit](#credit)

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
pip install -e .
```

Now setup your ClearML integration, go to your [ClearML Dashboard](https://app.clear.ml/dashboard) and run:

```shell
clearml-init
```

From there you will need to add AWS Access Keys to access models stored in S3.

## Training

Training can be completed using ClearML tasks or inside Jupyter Notebooks

## Jupyter Notebooks

These are found [here](notebooks/) and used for development, model training, and testing. All functionality should be migrated to the [amplify](amplify/) module.

## Deployment

This model has been packaged in a Dash application that can be deployed in a container.

### Dash App - Docker Compose

To deploy with Docker Compose, first make a file called `.env` and input keys/secrets for [OpenWeather](https://openweathermap.org/api/one-call-api), ClearML, and AWS.

```shell
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

```shell
docker-compose up -d --build
```

### Credit

Please give credit to the authors of this project if you use it in your own work. Reach out to us if you have any questions!

---

Made with ⚡ by [John](https://www.linkedin.com/in/john-droescher/), [Christian](https://www.linkedin.com/in/christianwelling/), and [Sam](https://samsipe.com)
