---
toc: true
comments: true
image: images/article3.jpeg
layout: post
description: Understanding the workings of scikit-learn’s ‘eli5’ library to compute feature importance on a sample housing dataset and interpreting its results
categories: [markdown]
title: How to Use scikit-learn ‘eli5’ Library to Compute Permutation Importance?
---

### Feature Permutation Importance with ‘eli5’ | [Towards AI](https://towardsai.net/)

## How to Use scikit-learn ‘eli5’ Library to Compute Permutation Importance?

### Understanding the workings of scikit-learn’s ‘eli5’ library to compute feature importance on a sample housing dataset and interpreting its results

![cc: Forbes](https://cdn-images-1.medium.com/max/2000/1*U1LPRoodN_CLmwq2SnFhPQ.jpeg)

Most of the Data Scientist(ML guys) treat their machine learning model as a black-box. They don’t know what are the things which are happening underhood.
 They load their data, do manual data cleaning & prepare their data to fit it on ml modal. Then the train their model & predict the target values(regression problem).

**But they don’t know, what features does their model think are important?**

![](https://cdn-images-1.medium.com/max/2000/1*TBnt_U1s-X_f9Eham_plCQ.jpeg)

For answering the above question Permutation Importance comes into the picture.

## What is it?

Permutation Importance is an algorithm that computes importance scores
for each of the feature variables of a dataset,
The importance measures are determined by computing the sensitivity of a model to random permutations of feature values.

## How does it work?

The concept is really straightforward: 
We measure the importance of a feature by calculating the increase in the model’s prediction error after permuting the feature. 
A feature is “important” if shuffling its values increases the model error because in this case, the model relied on the feature for the prediction.
A feature is “unimportant” if shuffling its values leave the model error unchanged because in this case, the model ignored the feature for the prediction.

## Should I compute importance on Training or Test data(validation data)?

The answer to this question is, we always measure permutation importance on test data.
 permutation importance based on training data is garbage. The permutation importance based on training data makes us mistakenly believe that features are important for the predictions when in reality the model was just overfitting and the features were not important at all.

## eli5 — a scikit-learn library:-

eli5 is a scikit learn library, used for computing permutation importance.

### caution to take before using eli5:-

**1. **Permutation Importance is calculated after a model has been fitted.
 
 **2. **We always compute permutation importance on test data(Validation Data).

**3. **The output of eli5 is in HTML format. So, we can only use it in the ipython notebook(i.e Jupiter notebook, google collab & kaggle kernel, etc).

## Now, let us get some test of codes 😋

![](https://cdn-images-1.medium.com/max/2000/1*FKiXevC6N5GwuurghctByw.gif)

I’ve built a rudimentary model(RandomForestRegressor) to predict the sale price of the housing data set.
 This is a good dataset example for showing the Permutation Importance because this dataset has a lot of features.
So, we can see which features make an impact while predicting the values and which are not.

 <iframe src="https://medium.com/media/403caf5e064b28b365194f1d03ecf74d" frameborder=0></iframe>

*Now, we use the ‘eli5’ library to calculate Permutation importance.*

 <iframe src="https://medium.com/media/91c8acade31aa861b8d5e1be98f37473" frameborder=0></iframe>

*you can see the output of the above code below:-*

![](https://cdn-images-1.medium.com/max/2000/1*8uCcJc3BZrJ1QdIGXPpXDQ.png)

## Interpreting Results:-

Features have decreasing importance in top-down order. 
The first number in each row shows the reduction in model performance by the reshuffle of that feature. 
The second number is a measure of the randomness of the performance reduction for different reshuffles of the feature column. 
 overallQual(overall quality) feature of the housing data set makes the biggest impact in the model while predicting the Sale Price.

### *You can get the housing-data set in .csv format from my GitHub profile*
LINK:- [https://github.com/abhinavsp0730/housing_data/blob/master/home-data-for-ml-course.zip](https://github.com/abhinavsp0730/housing_data/blob/master/home-data-for-ml-course.zip)

### You can also get .ipnyb file(kaggle Kernel) file from my GitHub profile
LINK:-

### [https://github.com/abhinavsp0730/housing_data/blob/master/kernel659579854a(2).ipynb](https://github.com/abhinavsp0730/housing_data/blob/master/kernel659579854a(2).ipynb)

### THANK YOU

### If you enjoy my article then do claps and follow me ❤️.

![](https://cdn-images-1.medium.com/max/2000/1*HnhqbqJ1vlHFEZmO5oEtqQ.gif)
