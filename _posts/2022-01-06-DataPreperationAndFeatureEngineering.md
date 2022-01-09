---
layout: post
title:  "Data PrePeration and  Feature Engineering"
categories: Notes
background: '/img/posts/numpy/cheetSheet.png'
---

## Data preperation and feature engineering

[refrred from](https://developers.google.com/machine-learning/data-prep)  
__Why to learn this__
- Incorrect data can confuse the model, which lead to wrong prediction 
- simple model with large and quality data can lead to better prediction

__Process__
1. define an ML problem and propose a solution
2. __Construct your dataset__
    - Collect raw data
    - Identify feature and label sources
    - select a sampling stretegy 
    - Split the data
3. __Trasnform data__
    - Explore and clean your data
    - Feature Engineering
4. train a model 
5. use model to make prediction 




### Collecting Data

> Your friend Sam is excited about the initial results of his statistical analysis.
 He says that the data show a positive correlation between the number of app downloads and
 the number of app review impressions. But he's not sure whether they would have downloaded it anyway
 without seeing the review. What response would be most helpful to Sam?

- lets compare similar user set, of formar one who have seen the review and downloaded the app with the later one  Who have downloaded the app with out observing the reviews. Now compare the frequency of download, this will atleast make us sure that yes correlation is true.

1. __Size and Quality of dataset__
    - Simple models on large data sets generally beat fancy models on small data sets.
    - It’s no use having a lot of data if it’s bad data; quality matters, too. But what counts as "quality"?
    -  the data is good if it accomplishes its intended task.
    -  Certain aspects of quality tend to correspond to better-performing models:
        - reliability   
            - it means can you trust your data source. 
            - few points which helps to boost reliability on data are :
                - How common are label errors? For example, if your data is labeled by humans, sometimes humans make mistakes.
                - Are your features noisy? presence of outlier in features beacuse of faulty instrument.
                - Is the data properly filtered for your problem? Keep only relevent data in data set , like for spam detection you  need search queries by bots. but if we want to recomment search results , we dont need queries which bots use, instead we need voacabulary which human can use.
                
           - What can make a data unreliable 
               - missing data for any feature 
               - duplicate data
               - false label 
               - outlier in feature value. 
        - feature representation
           - How can you map your data to useful representation.
               - it mean in which format you are passing your data to model
               - Should you normalize numerical value
               - how to handle outlier
           
        - minimizing skew(not similar)
           - What is Skew ?
               - lets say you get good result in training and testing. but in live environment results are not holding up. __Where can be problem__.
               - This problem suggests training/serving skew—that is, different results are computed for your metrics at training time vs. serving time.
           - How to overcome this ?
               - Always consider what data is available to your model at prediction time.
               - During training, use only the features that you'll have available in serving, 
               - make sure your training set is representative of your serving traffic.
           - Example 
               - Suppose you have an online store and want to predict how much money you’ll make on a given day. Your ML goal is to predict daily revenue using the number of customers as a feature. What problem might you encounter? Click the plus icon to check your answer.
                > The problem is that you don't know the number of customers at prediction time, before the day's sales are complete. So, this feature isn't useful, even if it's strongly predictive of your daily revenue. Relatedly, when you're training a model and get amazing evaluation metrics (like 0.99 AUC), look for these sorts of features that can bleed into your label.

2. __Joining Data Logs__
     1. __type of data logs__
    
        - When assembling a training set, you must sometimes join multiple sources of data.
            - Transactional log
            - attribute data
            - aggregate statistics

        - Transactional logs
            - we always Record specefic event. like when/who, why  it happened
            - it record transactional event for example it contain ip address, date, time at which some specefic event hasbeen performed

        - attribute data
            - transactional data deal with specefic moment, it deal with range of time. 
            - attribute data and transactional log are related.
            - this is also not of specefic event, it contain informationa about event.
           
            - it contain demography of user/ search history of user .

        - aggregate statistics
            - create a single feature by aggregating  many transactional log
             - we can join multiple transactional log and create some aggregate feature, like active time of user, this can be created by aggegating and finding average login time.
            - frequency of user query
            - average click rate on certain ad

     2. __Joining log sources__

        - Leverage the user's ID and timestamp in transactional logs to look up user attributes at time of event.
        - Use the transaction timestamp to select search history at time of query.

        > It is critical to use event timestamps when looking up attribute data. If you grab the latest user attributes, your training data will contain the values at the time of data collection, which causes training/serving skew. If you forget to do this for search history, you could leak the true outcome into your training data!
        
        - 

    3. __Prediction of data source__
        - you can featch data in two mode 
        - online mode
            - latency is always a oncern. so system must generate the input quickly.
        - offline mode
            - You likely have no compute restrictions, so can do similarly complex operations as training data generation.

        >For example, attribute data frequently needs to be looked up from some other system, which could introduce latency concerns. Similarly, aggregated statistics can be expensive to compute on the fly. If latency is a blocker, one possibility is to precompute these statistics.

3. __Identifying Label and sources__
- labelling your data is always challanging. you have to do it manually. now let say i give you sample of email. and our objective is predictin weather its s spam or important

- the first thing is know your objective
- i can read attribute like mail subject. now if i look the past behaviour of user, he has treated such subject lines as spam.
- so that mail will be categorized as spam.

    1. There are two type of label 
        - direct label 
        - derived label 
        
        for example - objective - check if user is fan of actor
        - direct label : are you a fan of actor. 
        - indirect label : how many movies a user as seen of an actor.
        - how many clicks user has done on actor clip 
        
    2. label sources
        - Event - it is an action 
        - attribute- here we have to check the past behaviour
        
        - for event we can ask the user directly about there feed back 
        - for attribute we have to measure the past behaviour. (here we already need supervised learning for setting the data.)
         
    3.  What if You Don’t Have Data to Log?
    Perhaps your product doesn’t exist yet, so you don't have any data to log. In that case, you could take one or more of the following actions:

        - Use a heuristic for a first launch, then train a system based on logged data.
        - Use logs from a similar problem to bootstrap your system.
        - Use human raters to generate data by completing tasks.
        
        

## Sampling and Splitting of data

- During EDA you have to understand the distribution of your data why? 
    - it will helps to know if its uniform, time variant
    - to know if any outlier is present.
    - to understand the balancing of data, if it is balanced or skewed to certain labels.

- What to do if too much data is in hand.
    - you have to select features based on objective
    - for a particular feature you have to select the data events(row/data point). 
    - now if you check the detribution and found some outlier/ rare events . you can decide to filter them .
    - this filtering may cause generalisation error, as when ever your model in real world sees those rare events, it will predict them wrong.
    
- what to do if you have very less data.
    - go with heuristic first and then collect more data.
    
    
- What to do if you have imbalanced data set 
    - majority class
    - minority class
    - [How to deal with imbanalnced dataset](https://towardsdatascience.com/how-to-deal-with-imbalanced-data-34ab7db9b100)
        - up sampling 
            - adding duplicate entries
        - down sampling 
            - take random data from majority class
        - up-weighting 
            - adding weights to your majority class label after doing downsampling and when calculating accuracy use it.
            - take probability score instead.
            
        - change performance matric from accuracy . calculate confusion matrix and work on precision tp/all predicted positive , recall : tp/all actual positve , F1 score
        - use decision tree algorithm , 
    1. you can train your model with minority class. and if its predicts them right its is working .
   
- How to split 
    - Random split , if data is not changing over time
    - split based on time, example analyse news sample.
    
    - where not to do random sampling 
        - time serires
        > Random splitting divides each cluster across the test/train split, providing a “sneak preview” to the model that won’t be available in production.
         
        - grouping data
        > The test set will always be too similar to the training set because clusters of similar data are in both sets. The model will appear to have better predictive power than it does.
        
        - Data with burstiness (data arriving in intermittent bursts as opposed to a continuous stream)
        
        > Clusters of similar data (the bursts) will show up in both training and testing. The model will make better predictions in testing than with new data.
    
    - Need to create three sets
        - train
        - evaluate
        - test
        - we can use crosse validation in training set 
        - make sure , when we are doing vectorization of data, do it saperately for train and test data. even normaliation also .
        


```python
## Transform your data 

```
