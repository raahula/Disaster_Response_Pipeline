# Disaster Response Pipeline Project
## Table of Contents
* Installation
* Instructions
* File descriptions
* Results
* Author
* Acknowledgement

## Installation: 
This project is based on Python 3.7.0 (default, Jun 28 2018, 08:04:48) running in an anaconda 5.3.0 distribution.
Plotly if not already installed will need to be installed using the command 'conda install -c plotly plotly=3.5.0'

## Instructions:
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://localhost:3001/

## File descriptions:

The project consists of the following files, the folders and subfolders are also mentioned.
- app
  - template
    - master.html  # main page of web app
    - go.html  # classification result page of web app
  - run.py  # Flask file that runs app
- data
  - disaster_categories.csv  # data to process, provided by [Figure 8](https://www.figure-eight.com/)
  - disaster_messages.csv  # data to process, provided by [Figure 8](https://www.figure-eight.com/)
  - process_data.py  # cleans up the data and saves it to DisasterResponse.db
  - DisasterResponse.db  # database to save clean data to
- models
  - train_classifier.py  # ML part of the code
  - classifier.pkl  # saved model 
- README.md

## Results:
The model performs reasonably well to predict the message category based on the incoming message.

## Author:
### Rahul Dixit
https://github.com/raahula

## Acknowledgement: 
Thanks to [Udacity](https://www.udacity.com/) and [Figure 8](https://www.figure-eight.com/) for providing me the oppurtunity to work on this great project.