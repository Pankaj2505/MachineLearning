---
layout: post
title:  "Introduction to ML"
categories: Notes
background: '/img/posts/numpy/cheetSheet.png'
---



## Objective
__Definition__
Machine Learning changes the way you think about a problem. The focus shifts from a mathematical science to a natural science, running experiments and using statistics, not logic, to analyse its results.

## ML Process
1. Define a ML problem and propose a solution
    - Articulate your problem
    - See if any labeled data exist
    - Design data for model
    - Determines where data come from 
    - Determine easily obtained inputs
    - Determine quantifiable output
2. Construct your dataset
3. Transform Data
4. Train your model
5. Use model to make prediction

## ML  Terminologies
####  Common ML Problem
1. Supervised 
    - labeled training data
    - express pattern as function
        - classification 
        - regression 
        - structured output(image recognition)
        - ranking (search result ranking)
2. Unsupervised
    - unlabeled training data
    - we find clusters in data, and then labelled them .
        - Clustering
        - Association rule learning (recommandation)      
        
3. Reinforcement 
    - here machine will award some point when ever machine will predict correctly.
    
    


 
#### Data Set
1. Features (X)
2. Label (Y)
3. model - which determines the relationship between features and label.
4. bias - some pattern which we dont want to learn, incorrect pattern, error.

### How to handle a problem
1. Models mistake are difficult to debug. as it can be any of below reason
    - Skewed training data
    - unexpected interpretataion of data during training time.
    - users are handling product in other than prescribe way.
    
2. How to address challanges of transition to ML

|Step                                     | 	Example                                 |
|-----------------------------------------|---------------------------------------------|
|1. Set the research goal.|	I want to predict how heavy traffic will be on a given day.|
|2. Make a hypothesis. |	I think the weather forecast is an informative signal.|
|3. Collect the data.	|Collect historical traffic data and weather on each day.|
|4. Test your hypothesis.|	Train a model using this data.|
|5. Analyze your results.|	Is this model better than existing systems?|
|6. Reach a conclusion.	|I should (not) use this model to make predictions, because of X, Y, and Z.|
|7. Refine hypothesis and repeat.|	Time of year could be a helpful signal.|

__challanges in ML problem__
1. What is the problem , then check how ML can help you.
2. Exploratory data analysis can help you understand your data.
3. collect the data, thousand of sample for ML models and hundreds of thusand of sample for neural networks.
4. Predictive power depends on features.
    - You should not try to make ML do the hard work of discovering which features are relevant for you. If you simply throw everything at the model and see what looks useful, your model will likely wind up overly complicated, expensive, and filled with unimportant features.
    - perform inferential statistics for smaller datasets.
5. Aim to make decision , from the prediction. 
    - like recommanded system once recommend something, you have to provide the link for recommendation.
    - show ads only when probability of click is greater than threshold.
6. define cluster name 
7. what will happen if a new data comes in production what to do then.
8. ML models are not always good, heauristic(if /else ) approach can fix anamolies detection easily
9. there is diffrence between corelation and causation
    - coorelation mean , how one feature is changing according to the change in other feature
    - caustion mean , change is one fature is causing change in other feature.
    
10 - Challanges in data gathering
> Your model can make predictions at either of two points:  
        In real time, in response to user activity (online).  
        As a batch and then cached (offline).  
        
    - What data does your code have access to when it needs to call the model?
    - What are your latency requirements? Do you need to run quickly to avoid lagging in your UI, or are you running without a user waiting for your model?
    -  be wary of using out-of-date data. 

__Approach for implementing ML__
1. Objective - what would you like your ML model to do?
    some time a related but indirect goal can help to realise main goal.
    - We want the ML model to predict how popular a video just uploaded will become in the future.
    - indirect goal - predict the share count , and predict the watch time.
    
2. Your IDEAL outcome Decision - what should your model do after predicting, what is the outcome.
    - here we want to recommend only those videos which are worthtime of users.
    
3. Design an approach-(look for feature needed for trainng)
    - for above problem , we want to collect "category of video" and there "watched time", "click count".
    
4. Success or failure metric 
    - define which metric you will choose
        -  precision /recall/AUC(Area under ROC curve)(Receiver operating chracterstic curve).
    - quantify the metric 
        - how much presion value mean success/failure.
    - you should know other factors which can fail the model
        - availabilty of data to model
        - latancy in data availbilty
        - very old data is trained
5. you should be aware which ML algo to use for what type of problem

|Type of ML Problem	|Description|	Example|
|-------------------|-----------|----------|
|Classification	|Pick one of N labels	|cat, dog, horse, or bear|
|Regression	|Predict numerical values	|click-through rate|
|Clustering	|Group similar examples	|most relevant documents (unsupervised)|
|Association rule learning	|Infer likely association patterns in data	|If you buy hamburger buns, you're likely to buy hamburgers (unsupervised)|
|Structured output	|Create complex output	|natural language parse trees, image recognition bounding boxes|

6. you should know what to do with predicted outcome
    


## Exercise1 
![project link](r"C:\Users\04136O744\Desktop\DataScience\MachineLearning\image\how to approach ML problem.pdf") 

## Formulate your ML problem
This section is a guide to the suggested approach for framing an ML problem:

1. Articulate your problem.
    - supervised
        - classification 
            - binary class classificaiton 
            - multi class classification
                - Multiclass single label(which animal is in picture)
                - multiclass multilabel (all animal in pictture)
        - regression 
            - uni-dimensional regression ( hight of player)
            - multi-dimensional regression (predict longitude/latitude)
    - unsupervised
        - association rule(recommending similar type)
        - clustering
    - reinforcement
    
    
2. Start simple.
    - Simplify your modeling task means try to do it using binary classification or uni dimensional regression
    - apply the simplest algorithm possible, as you will know from the scratch if simple theorem can fix this or not.complex models are hard to debug.
    - start from only 1- 3 features to do the prediction .

3. Identify Your Data Sources.
    - how much labelled data you have
    - what is the source of labelled data.
    - are labelled closelly related to the decision you are trying to make(not prediction)
    
4. Design your data for the model.
    - find the best possible features and label.
    
5. Determine where data comes from.
    - Assess how much work is needed to develop a pipeline.
    - assess what will happen in case output from pipeline is available.

6. Determine easily obtained inputs.
    - use 1-3 feautres which are available.
    
7. Ability to Learn.
    - __unbalance dataset__ The data set doesn't contain enough positive labels.
    - __less datapoint__ The training data doesn't contain enough examples.
    - __outlier__ The labels are too noisy.
    - __baised error__ The system memorizes the training data, but has difficulty generalizing to new cases.



```python

```
