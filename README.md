# NFL Game Winner Prediction

---

## Logistic Regression from Scratch

HOW TO RUN
----------
Step 1 - Install dependencies:

    pip3 install nflreadpy pandas numpy matplotlib pyarrow

Step 2 - Generate the dataset (only needs to be done once):

    Run the nfl_scraper.ipynb notebook

    This pulls data from nflverse for the 2022-2025 seasons and saves
    it as nfl_data.csv in the data folder.

Step 3 - Run the model:

    python3 logistic_regression.py

    Make sure nfl_data.csv is in the data folder, which should be in the same directory as the program file.
    The script will automatically find it.

Step 4 - Check the outputs:

    - Results print to the terminal (learning rate comparison,
      confusion matrices, accuracy/precision/recall/F1).
    - lr_training_curves.png is saved to the current folder.
    - lr_feature_importance.png is saved to the current folder.

---

## Naive Bayes Model

HOW TO RUN
----------
Step 1 - Install dependencies:

    pip3 install nflreadpy pandas numpy

Step 2 - Generate the dataset (only needs to be done once):

    Run the nfl_scraper.ipynb notebook

    This pulls data from nflverse for the 2022-2025 seasons and saves
    it as nfl_data.csv in the data folder.

Step 3 - Run the model:

    python3 naive_bayes.py

    Make sure nfl_data.csv is in the data folder, which should be in the same directory as the program file.
    The script will automatically find it.

Step 4 - Check the outputs:

    - Results print to the terminal (confusion matrices, accuracy/precision/recall/F1, class priors).

---

## Random Forest Notebook

HOW TO RUN
----------
Step 1 - Install dependencies

    pip3 install nflreadpy pandas numpy matplotlib plotly

Step 2 - Generate the dataset (only needs to be done once):

    Run the nfl_scraper.ipynb notebook

    This pulls data from nflverse for the 2022-2025 seasons and saves
    it as nfl_data.csv in the data folder.

Step 3 - Run the notebook

    Make sure nfl_data.csv is in the data folder, which should be in the same directory as the program file.
    The script will automatically find it.

    This will automatically provide output as each code cell runs, with validation accuracies and feature importances at the end of the notebook