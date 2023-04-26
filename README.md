Project Instructions
==============================

This repo contains the instructions for a machine learning project.

# **Title:** Estimating Earnings After Graduation from Post-Secondary Institutions

## Project Abstract

The process for any student who is planning on attending university or college after highschool is an overwhelming one. With so many different post-secondary institutions available for students, narrowing down their potential future school can be a difficult task. One major factor when applying to schools is how successful a student will be after graduation. Knowing what a students outcome may be after attending a specific institution can help guide their decision and ultimately choose a post-secondary institution that helps them reach their academic and career goals.

The primary objective of this project is to create a tool that provides a good estimation of a students future earnings based off the characteristics of an institution. A secondary objective of this project is to find the important and interesting features of a school that correlate with higher/lower future earnings. Students will be able to utilize this project to aid in their search for the post-secondary institution that meets their standards and institutions will be able to look at the data gathered from this project and see what features correlate to increasing the outcome of their students.

To build this tool, machine learning techniques will be utilized to create a model that provides an accurate estimation of future earnings based on the features of a school. Multiple machine learning models will be used on the data and the results will be compared to find the best estimator. The models that will be considered are: SVM, Decision Trees, K-Nearest Neighbours, and Lasso. Additionally, other machine learning techniques will be used on top of these models to improve results further, such as bagging and boosting. Finally, feature selection will be performed to extract the features that are the most important in determining future earnings, these results will further aid in understanding what is important to consider when comparing schools and finding the best school given a students circumstance.   

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for describing highlights for using this ML project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   └── processed      <- The final, canonical data sets for modeling.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │   └── README.md      <- Youtube Video Link
    │   └── final_project_report <- final report .pdf format and supporting files
    │   └── presentation   <-  final power point presentation 
    |
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
       ├── __init__.py    <- Makes src a Python module
       │
       ├── data           <- Scripts to download or generate data and pre-process the data
       │   └── pre_processing.py
       │
       ├── features       <- Scripts to turn raw data into features for modeling
       │   └── build_features.py
       │
       ├── models         <- Scripts to train models and then use trained models to make
       │   │                 predictions
       │   ├── predict_model.py
       │   └── train_model.py
       │
       ├── visualization  <- Scripts to create exploratory and results oriented visualizations
       │   └── visualize.py           
       └── main.py        <- Main script used to run the program and call all the other scripts

## Using The Project

### Running the Project
- Make sure all the required packages are installed into your environment
- Open up a terminal
- Navigate to the *src* folder
- Run the python file *main.py*

Note: Due to the way pathing is set up, if you do not run the program from within the *src* folder, the program will not work.

Some things to consider before using the project:
- To use the program run the python module src\main.py, from there a CLI interface will appear and present options to select.
- The choices that are presented when running main.py are in order of how the project should be executed if running from scratch, however after the dataset has been prepared it can be used again without having to go through all the preprocessing.
- The dataset in the \data\external directory is compressed due to its size, but when main.py is called it will be unzipped once the user selects the option to clean the dataset.
- The SVM model takes upwards of 10 minutes to train, and takes a few minutes to predict.
