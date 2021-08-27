![cover](./images/New-York-Skyline.jpg)

# NYC Airbnb Analysis
**Author**: Jennifer Ha

## Overview
This project analyzes New York City Airbnb listings data for the last 12 months (August 2020 - July 2021), which includes detailed information about the hosts and the listings. For the purpose of this project, Airbnb marketing team would like know which listings are valuable enough to be selected for their NYC promotion page in the coming winter. With the strong vaccination rate and eased Covid-19 regulization in the city, the team expects that more visitors would be looking for a place to stay in NYC moving forward. Since they cannot go over and compare each listing one by one, the team is looking for a prediction model that they can rely on to carefully select valuable listings that they can promote. They further anticipate to rely on the selected model from this analysis for their future promotions as well.

To help the Airbnb marketing team with accurately selecting the valuable listings in NYC, I'll be comapring classification models and further optimize the best perofrming model.
***
## Business Problem
The goal of this analysis is to predict whether a listing is valuable or not. The Airbnb marketing team has decided to consider top 25% listings as valuable, and have asked to calculate a weighted review scores rating to reward listings with more number of reviews and penalize listings with less number of reviews.

The team should be able to use the best model from this analysis to decide which listings are valuable and should be promoted to people who are searching for a place to stay when they visit New York City. We will focus on accuracy, precision, and ROC-AUC scores to determine which model performs the best. In this problem, accurately identifying a valuable listing is important as we cannot false advertise a listing with poor rating.
***
## Data
The dataset has 74 columns and 36722 rows of New York City Airbnb listings data from August 2020 - July 2021. Each feature contains detailed information about the host and the listing, and we will be exploring each of them to decide which relevant features to keep for this project. The final DataFrame we will be working with will have 16 feature columns with the same number of rows as we remove IDs, URLs, repetitive information.

The target variable we would like to predict is `weighted_review_scores_rating`, which we will calculate using the number of reviews and review scores rating provided in the original dataset. Then, we will convert it to binomial, 0 = No (not valuable), 1 = Yes (valuable). Class imbalance is expected since we will consider the top 25% of the listings as valuable.
***
## Methods
This project explores 5 different machine learning model types using the SKLearn package: logistic regression, K-Nearest Neighbors, Decision Tree, Random Forest, and Adaboost. Due to overfitting problem, I've decided not to move forward with K-Nearest Neighbors, Decision Tree, and Random Forest models. To further improve the performance, hyperparameter tuning was performed on logistic regression and Adaboost models.

***
## Results



## Conclusions



***
## For More Information
See the full analysis in the [Jupyter Notebook](https://github.com/jennifernha/NYC-Airbnb-Analysis/blob/main/NewYork-Airbnb-Analysis.ipynb) or review this [presentation](https://github.com/jennifernha/NewYork-Airbnb-Analysis/blob/main/Presentation.pdf).
For additional info, contact Jennifer Ha at jnha1119@gmail.com
***
## Repository Structure
├── data 
├── images                        
├── New-York-Airbnb-Analysis.ipynb   
├── Presentation.pdf                   
├── README.md                                    
└── functions.py 
  