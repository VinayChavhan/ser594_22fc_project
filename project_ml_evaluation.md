#### SER594: Machine Learning Evaluation
#### Identification of Defects in Silicon Wafers
#### Vinay Chavhan
#### 11/21/2022

## Evaluation Metrics
### Metric 1
**Name:** Confusion matrix

**Choice Justification:** We are solving the classification problem where we want to know the category of the image. We have nine categories of defects like ['Center' 'Donut' 'Edge-Loc' 'Edge-Ring' 'Loc' 'Near-full' 'Random' 'Scratch' 'none'] and we have images. Now we want to see that in prediction for all categories how many images are rightly categorized?

Interpretation: ** We are using the XGBoost as our final model cause it’s giving the highest accuracy which is close to 99%, For that algorithm, we are getting the matrix below.
[[ 466    0    2    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0    0    0]
 [   0    0  119    0    0    0    0    0    0]
 [   0    0    0    6    0    0    0    0    0]
 [   0    1    0    0   81    0    0    0    0]
 [   0    0    0    0    0    6    0    0    0]
 [   0    0    0    0    0    0   10    0    0]
 [   0    0    0    0    0    0    0   17    0]
 [   2    0   11    0   12    0    1    5 5498]]

If we observe that for category 0 which is the center, 466 images were rightly predicted and 2 images were wrongly predicted. Likewise, this is a very good metric to show the model's performance.


**Name:** F1 score

**Choice Justification: **  F1 score gives a good balance between precision and recall, whenever both values are low then the F1 score would also be low. It is also called the F1 measure. The F1 measure is an effective way to get the classification model performance. The F1 score is calculated below
F1 = 2 [(Recall * Precision) / (Recall + Precision)]
But we can also calculate the F1 score from the confusion matrix that we have generated already using the below formula.
F1 = (True Positive) / [True Positive + 1/2*(False Positive + False Negative)]
A high F1 score means our model performed well cause it has both high values of precision and recall.

Interpretation: ** The final model XGBoost gives us the F1 score of 99.45, which is almost 100. This indicates that our final model is performing very well.

## Alternative Models
### Alternative 3

RandomForestClassifier

**Construction:** Random forest classifier has the below parameters needed when constructing the model, n_estimators: int, default=100, and train_X in an array-like shape and train_Y in an array-like shape. There are other parameters but we are not passing them and the model is using default values. If we pass the n_estimators = 1000 then the model takes a lot of time to build and accuracy comes around the 95-99 percent range. But if we reduced the n_estimators to 100 then model builds is a good time and give accuracy in the same range.  We have split the data into 80 and 20 with respect to train and testing and applying it to the model.  If we relate this model to our final model which is XGBoost then we can see that the internal working of the XGBoost, Gradient boosting is a supervised learning algorithm, which attempts to accurately predict a target variable by combining the estimates of a set of simpler, weaker models.

**Evaluation:** The Random forest algorithm gave 97.27 percent accuracy. The Root means the squared error is 72.35 and the F1 score is 97.27. This model also performed well looking at the evaluation matrices. The confusion matrix given in the summary.txt also looks good


Support vector machine(SVM) Classifier

**Construction:** SVM classifier is a supervised learning model with associated learning algorithms used for classification as well as regression. If it gets the labeled training data then the algorithm outputs an optimal hyperplane that categorizes the new samples. In our case we want the same, we need to give the model training data and test our 20 percent data and the model should identify the label correctly. We are considering only 9 defects right now. We only have one feature which are images that we are feeding to the algorithm, the accuracy of the XGBoost and SVM is coming almost the same, there is not much of a difference as per our model but we can see that SVM taking a lot of time to build which is also longer than the XGBoost model building. So if we relate both classifiers then we can say that XGBooost is better.

**Evaluation:** The model showed the accuracy of 96.95 and F1 score of 96.95. We can deduce that the model is performing well by considering only these two measures. We also have a confusion matrix and root mean squared as 72.48. The observation confusion matrix is, the category of 0(‘center' defect), the model predicted all that category images correctly which are 443.


KNeighborsClassifier

**Construction:** KNN is a very interesting algorithm, it calculates the distance between all the points in the plane and checks the most common of them all, and gives itself a label. Here we have passed the k value as 4 then each point will check its nearest four neighbors and select the most common label they have. The model building happens very quickly. So, it does not take much time to build and give the results which are good. The KNN uses the nearest neighbors and does not check all the points. This is the main reason that its accuracy is lower than other models. If we relate this to our main model which is XGBoost then we can see that XGBoost has higher accuracy cause it considers all the points and the algorithm uses Gradient boosting is a supervised learning algorithm, which attempts to accurately predict a target variable by combining the estimates of a set of simpler, weaker models.

**Evaluation:** The model showed an accuracy of 88.39,  and an F1 score of 88.39. This shows that model did not perform well compared to other models that we checked. The root mean squared is 227.06 and if we observe the confusion matrix in summary.txt, we see that lots of predictions are wrong.



## Visualization
### Visual N
**Analysis:** 

Confusion metrices heatmap 
1)visuals/KNeighborsClassifier_confusion_metrix.png
2)visuals/RandomForest_confusion_metrix.png
3)visuals/SVC_confusion_metrix.png
4)visuals/XGBoost_confusion_metrix.png
5)visuals/type_of_defects.png



## Best Model

**Model:** XGBoost performed well in all the models with an accuracy of 99.49 which is higher than other alternate applied models. Also, Boost is taking less space when we save the model for future use. The F1 score is 99.45 and  Root mean squared is 11.89 which is lower and better. If we observe the confusion matrix resent in summary.txt then we can see that very minimal wrong predication is present which is good.