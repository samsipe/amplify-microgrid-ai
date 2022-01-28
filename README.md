# Amplify Microgrid AI

## Overview
**This project aims to modernize how buildings interact with the grid**, utilizing deep learning techniques to predict future usage and generation to make smarter energy decisions.

### Project Details
[Google Drive](https://drive.google.com/drive/folders/1sVjw4bLe3xxM489szpL0qAIXmHHzE7Xp?usp=sharing)

## Setup and Installation
Run the following command to get `amplify` installed in your environment.

```shell
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Now setup your ClearML integration, go to your [ClearML Dashboard](https://app.clear.ml/dashboard) and run:
``` shell
clearml-init
```

## Training
CLI functionality has been added to this python module so you can run the following.

``` shell
amplify --version
```

## Jupyter Notebooks

These are found [here](notebooks/) and  used for development and testing. All functionality should be migrated to the [amplify](amplify/) module.
