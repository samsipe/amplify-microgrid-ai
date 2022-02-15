# Amplify Microgrid AI

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

## Dash App
To deploy the Dash app with Docker Compose, first make a file called `.env` and input an [OpenWeather API key](https://openweathermap.org/api/one-call-api) like this:

```
OW_API_KEY=this_is_an_api_key
```
Then run the following:

```
docker-compose up -d --build
```

This relies on the same `clearml.conf` file before with a docker bind mount. Make sure the absolute path is correct.
