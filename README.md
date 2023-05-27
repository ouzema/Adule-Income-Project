# Adule Income Project
 Machine Learning to predict adult income
# Contents
[I.	INTRODUCTION	](#_toc134400167)

[II.	Proposed Methodology	](#_toc134400168)

[1.	Dataset	](#_toc134400169)

[2.	Feature Engineering and Selection	](#_toc134400170)

[3.	Data Preprocessing	](#_toc134400172)

[i.	Handling Missing Values:	](#_toc134400173)

[ii.	Categorical Feature Encoding:	](#_toc134400174)

[iii.	Train Test Split:	](#_toc134400175)

[5.	Data Modeling	](#_toc134400176)

[III.	Results	](#_toc134400177)

[*IV.*	*Conclusion*	](#_toc134400178)
























I. <a name="_toc134400167"></a>INTRODUCTION

Since the advent of machine learning algorithms, the field of artificial intelligence has grown significantly. These algorithms have been applied to a variety of tasks, such as classification and regression research. Data mining and machine learning fields have investigated particular obscure patterns and ideas that have enabled the prediction of challenging future events, in addition to using them for research and discovery. In this experiment, a machine learning model was built from scratch in order to analyze data, extract or select its features, deal with missing values, and evaluate the effectiveness of various algorithms. Age, workclass, fnlwgt (final weight), education, education-number, marital status, occupation, relationship, race, gender, capital-gain, capital-loss, hours-per-week, native-country and income are the 15 attributes that make up the dataset.







II. <a name="_toc134400168"></a>Proposed Methodology
1. <a name="_toc134400169"></a>Dataset

A well-liked dataset for machine learning and data analysis is the Adult Income dataset, also called the "Census Income" dataset. The dataset includes details on almost 32,000 people from the 1994 US Census, including information on their age, education, job class, marital status, occupation, relationship, race, and sex, as well as information on their race and place of origin.

Income, which shows whether a person's income is larger than $50,000 per year, is the dataset's target variable. Based on the provided attributes, the dataset is frequently used to create classification models that determine whether a person's annual income is $50K or above.

The dataset includes categorical and numerical variables, and missing values are indicated in the dataset by a question mark (?). With the aim of predicting the target variable based on the provided features, the dataset is frequently used for supervised learning tasks like classification.

The Adult Income dataset has been utilized in numerous research studies and is a frequently used benchmark dataset in the machine learning community. Due to the inclusion of missing data, categorical features, and imbalanced classes, it presents a difficult problem for machine learning methods.







2. <a name="_toc134400170"></a>Feature Engineering and Selection

With feature-to-feature correlation between the attributes, which are all continuous variables, a correlation matrix is shown as a heat map in Fig. 1. The attributes education and education-num are most closely related, with a 90% similarity, as can be seen in the heatmap. To reduce the duplication of feature values, or to conduct an omission over the dataset, the attribute education-num is thus eliminated.







3. <a name="_toc134400172"></a>Data Preprocessing

Before processing the data using the Adult Income Dataset, the data must first be cleaned using pretreatment methods.

i. <a name="_toc134400173"></a>Handling Missing Values:

Since there are a few missing values (a few '?'s), the dataset's 48,842 records are used to represent them. 

Three attributes, namely workclass, occupation, and native-country, have such missing values. The most frequent value was used in place of the question mark because these are categorical values.

ii. <a name="_toc134400174"></a>Categorical Feature Encoding:

Few categorical attributes might be more closely clustered together and encoded for better representation. The label encoding used for these attributes is shown in Table 1.
















|**Feature**|**Grouped Value**|**Label Encoded Value**|
| :-: | :-: | :-: |
|Income|<=50k|0|
||>50k|1|
|Education|Preschool,1st-4th,5th-6th,7th,8th,9th,10th,11th,12th|0|
||HS-grad|1|
||Assoc-voc,Assoc-acdm,Prof-school,Some-college|2|
||Bachelors|3|
||Masters|4|
||Doctorate|5|
|Hours-per-Week|< 20|0|
||20 – 40|1|
||`         `> 40|2|
|Marital-Status|Married-civ-spouse,Married-AF-spouse|0|
||Never-married, Divorced,Separated,Widowed, Married-spouse-absent|1|
|Gender|Male|0|
||Female|1|


iii. <a name="_toc134400175"></a>Train Test Split:

Seventy percent of the dataset is made available for training, while the remaining thirty percent is divided into training and test sets.

5. <a name="_toc134400176"></a>Data Modeling

For the Adult Income dataset's categorization objective, a number of machine learning methods were employed. These include Support Vector Classifier, Random Forest Classifier, Logistic Regression, and KNeighbors Classifier. 

Grid Search is an estimator search technique that was used to find the ideal hyper-parameter for these algorithms. The optimal parameter for each model is shown in Table 2 along with the combination of hyper-parameters, train accuracy, and test accuracy.


|Classifier|Hyper-parameter|Best parameter|Accuracy|
| :-: | :-: | :-: | :-: |
||||Train|Test|
|KNeighbors Classifier|'n\_neighbors': range(1, 20, 2)|'n\_neighbors': 10|75\.6%|73\.9%|
|Logistic Regression|<p>'C': [1, 0.1, 0.01, 10, 100]</p><p>'penalty': ['l1', 'l2',</p><p>` `'elasticnet']</p>|'penalty': 'l2', 'C': 10|75\.2%|75%|
|Random Forest Classifier|<p>'criterion': ['gini', 'entrop']</p><p>'max\_depth': [1, 10, 5],</p><p>'min\_samples\_split':</p><p>[1, 10]</p>|<p>'min\_samples\_split': 10,</p><p>'max\_depth': 1,</p><p>'criterion': 'entropy'</p>|87\.9%|69\.3%|
|SVC|<p>'C': [1, 0.1, 0.01], 'kernel': ['linear’, ‘poly']</p><p></p>|<p>'C': 1,</p><p>'kernel': 'linear'</p>|75\.2%|75%|






III. <a name="_toc134400177"></a>Results

The dataset contains 48,842 occurrences, of which 31,655 were utilized for training and 13,567 were reserved for testing. RFC surpassed all other machine learning algorithms with a test accuracy of 75%. The following measures are used to access the model performance:

- Precision is calculated by dividing the actual true prediction by the model's total number of predictions.

*Precision = TP/(TP+FP)*

RFC model resulted with precision of 0.87.

- The recall is determined in a classification problem with two classes by dividing the total number of true positives by the sum of true positives and false negatives.

*Recall = TP/(TP+FN)*

*RF*C model resulted with recall of 0.69.

- A weighted average of recall and precision is the F1 score.

*F1 score = 2\*(Recall \* Precision) / (Recall + Precision)*

RFC model resulted with f1-score of 0.76.

- The classification model's success in correctly predicting examples from different classes is summarized in a table called the confusion matrix.





IV. <a name="_toc134400178"></a>*Conclusion*

The study found that RFC was more accurate than the other algorithms discussed. The confusion matrix identified 9.46% of the samples as false positives and 5.33% as false negatives. Therefore, the model's faults account for 25% of the test data. Further findings obtained 88.16% validation accuracy on an 80-20 train-test split, suggesting that further expanded versions of the present dataset can be employed to increase the model's accuracy. In order to get better outcomes overall while retaining accuracy, further work on this project will concentrate on integrating hybrid Artificial Intelligence approaches, which combine Machine Learning with Deep Learning (Neural Networks).
