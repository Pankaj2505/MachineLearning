---
layout: post
title:  "ML Intution"
categories: Notes
background: '/img/posts/numpy/cheetSheet.png'
---
#### Table of Contents

- [ML vs Heuritics](#ML vs Heuritics)
- [Train VS Test](#Train VS Test)
- [Center](#center)
- [Color](#color)

## ML vs Heuritics

- __Heuritcs__   
    - it is a set of instruction which a program executes to get the decision.
    - this will not change according to change in data.
    
    
- __ML__
    - ___WHAT___  here we teaches the program , we provide the data and corrosponding label, ML algo will learn in a way where it find the pattern, that for below features of data, our data belongs to certain category . and later on . when ever machines sees any new data points with similar set of feature, it predicts its class label.
    
    - ___ERROR___ this is calculated by finding difference between actual output and predicted output. smaller the error better the model.
    
    - ___ADVANTAGE___ as environment get updated, data is changed , we can retrain our model.
    



## Train VS Test
- the more data , better the model
- find the label - split data vertically to X, y
- split data horizontally into two part called  test and train.
- model.fit (X_train,y_train)
- y'_test = model.predict(Xtest)
- error = y-y'
- accuracy(y_test, y'_test)

## Overfitting Vs underfitting

- ___Overfitting___
    - when our model tries to cover all points of a dataset.
    - it will create confusion in model , and this leads to wrong prediction.
    - here we have given so many features, now confusion will arise
    
- ___Underfitt___
    - model is not covering majourity of a data set.
    - model is not at all predicting accurately. as it is not covering those feature sets.
    - it happens when we give less no of feature and datapoints to train the model
    
- Example 
    - we need to predict any round shape as ball
    - if we are passing only one feature called shape. model will become underfit and it will predict every round shape as ball. even fruits(Underfitting)
    - now we give features like , round, play= yes , eat= No, radious < 5cm.  now i have given football. the fourth condition will get failed.(Overfitting)

## Feature Selection
- ___Why___
    - useless features used in training will cause overfitting, training time will increase, complexity of model will increase.
    - one relevant features can imporove the model drastically , and it will make model very less complex


- ___How___
    - __Filter Method__(check relavance )
        - Information gain
           
        - Chi -Squre test
        - Co-relation Coefficient
            - we check how frquently lable is varying with changes in feature.
            - exmaple , roll no vs prediction of result
            - study hours vs prediction of result
    - __Wrapper Method__(check usefullness)
        - Recursive feature seletion
             - here we will pick and send all feature one by one to model, and measure the accuracy for each feature.
            - which ever feature has highest accuracy , we will pick it and send other features as a group of two. do this  with all feature and measure the accuracy . 
            - like wise continue the process , upto your satisfaction of accuracy. 
        - Recursive feature elemination
        - Generic algorithm
    - __Embedded Method__(Overfitting is very less)
        - decision Tree

## Model Hierarchy

- supervised (class label given)
    - classification 
        - binary 
        - [multi_class](https://towardsdatascience.com/multi-class-classification-one-vs-all-one-vs-one-94daed32a87b)
            - one vs all
            - one vs one
    - regression 
- un Supervised(label not give)
    - clusterring
- reinforcement(reward points)


__Multiclass classifier__  
1. One Vs All(one vs rest)
    - lets say we wanna predict on three fruits (apple, orange, banana)
    - genreate as many classifier , as much we have classes
    - if i have three class, i need to create three classifier, and for this i need to create three train dataset.
    - for 1st data set, we will predict is it orange or not so we will re-write the label , and keep orange as 1 and rest as 0
    - 2nd data set we will predict its apple or not so we will re-write the label , and keep apple as 1 and rest as 0
    - 3rd data set we will predict its banana or not so we will re-write the label , and keep banana as 1 and rest as 0
    - Now test tuple will be assign to all three model , which ever model proababilty prediction is greater we will assign that result.
    
2. One vs one 
    - we need to generate n*(n-1) / 2 classifier. it mean we will create as many pairs of classes for a multiclass label.

## Feature Engineering 

- Dimensionality Reduction 
    - PCA
    - TSNE
    
- Vectorization
    - Numerical 
        - Normalization
        - standardization
    - Categorical 
        - one-hot encoding
    - oridinal
    - Textual 
        - BAW
        - TF/IDF
        - AVG-TF/IDF
        
        
__PCA- Principal Componant Analysis__  
- ___Why___
    - to handle overfitting as too many features are present in model
    - how are we reducing the dimension(features),here we will try to find a new eign-vector, with new eign values by changing the views (eign- vector)
    - here we find principal componant(Eign vector), no of PC are less than or equal to no of attribute.
    - the best eign vector is always 1
    - PC1 and PC2 are always orthagonal , mean independent with each other.
    
- ___How___
    ![first calcualte covariance matrix](images/_1_PCA_part1.png "first calcualte covariance matrix")
    ![determinant eign value eign vectore](images/_2_PCA_part2.png)

## Data pre- processing

1. Data Cleaning
    - Handle outlier
    - Handle Duplicates
    - Handle Missing value
    - Handle Time series
2. Analysis of features
    - Check distribution of data how much its is skewed
    - analyse how each feature is identifying the class label
    - analyse how features are distributed to understand the linearity. this will help to choose best model by knowing if its linearly saperable or not.
3. statistical analysis
    - univariant 
        - we use charts to understand the distribution of single feature for stattistical analysis.
    - Bi variant
        - we draw charts using feature with label, to understand the variance and distribution

##  Generalization Error
- Bias variance trade-off
- Overfitting 
    - __What__
        - model is biased .
    - __How to fix__
        - reduce dimension
            - dimansionality reduction 
            - feature selection
- Underfitting
    - __What__
        - model has variace .
    -__How to fix__
        - add more no of datapoints
        - add more relevant features

## SVM(Support vector machine)
1. Supervised learning
2. used for both classification and regression ?
3. will it work for Linear and non linear data set ?

4. Here we create a __decision boundary__ or hyper plane. one side of this boundary is class 1 and other side is class 2
5. we will draw two more plane which are parallel to hyper palne and passing thorugh the nearest -ve class poin and +ve class points.
6. support vectors are those points from which the scondary lines are passing.
7. d+ and d- is the distnace between the hyper plane and support vector
8. sum of d+ and d- is called range.
9. ___Hyper plan is that plane where the range is maximum.___
10. how to use SVM in Non linear data 
    - kernal function is used to deal with non -linear dataset
    - kernal function takes low dimensional feature space and convert into high dimensional feature space.
    - low dimension features space mean , feature can not be divided using straight line
    - convert low D  to higher D space to easily saperate the data set 



```python

```
