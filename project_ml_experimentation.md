#### SERX94: Experimentation
#### Identification of Defects in Silicon Wafers
#### Vinay Chavhan
#### 11/21/2022
## Explainable Records
### Record 1
**Raw Data:**  We have processed data in a dictionary, the key of the dictionary is images and labels. We have 9 categories of defects which are ['Center' 'Donut' 'Edge-Loc' 'Edge-Ring' 'Loc' 'Near-full' 'Random' 'Scratch' 'none'] Please check visuals/type_of_defects.png to see the defects visualization. we will consider the row where the image defect is 'Center'. In our processed data row number 39 has an image for which the label is the center defect category. center defect means you will image will have blue points with a filled circle in the center.

Prediction Explanation:** We can see that model is giving the right output by looking at the confusion matrix. The algorithm is finding the output label by combining the estimates of a set of simpler, weaker models. 466 images were rightly predicted and only 2 were wrongly predicted.

### Record 2
**Raw Data:** As we explained before in record 1 we only have two columns in processed data. Now we will concentrate on the 'Edge-Loc' defect category.  The 0th-row image has this defect and visually you will see an image where the blue points are concentrated on any edge.

Prediction Explanation:** The model is predicting the values right and which are reasonable. Looking at the confusion matrix we can see that a higher number of images are identified correctly. The XGBoost model takes the estimation of the set of simpler and weaker models to predict the value and even the evaluation matrix accuracy and F1 score are high. We can logically say that the prediction is right and as expected.

## Interesting Features
### Feature A
**Feature:** Images (called “waferMap” in the original dataset)

**Justification:** This is the most main feature in the data set, whole processing is happening on this feature only, we will give the image to the model and predict the label for it which means the defect for it.

### Feature B
**Feature:** dieSize in the original dataset

**Justification:** This feature is present in the original dataset, which is not usable by the model. But this feature also looks valuable to consider after the Images feature in processed data.

## Experiments
### Varying A
**Prediction Trend Seen:** If we vary the Images feature then we can see that trend is changing, this is obvious that cause whole prediction is based on this feature. The model is taking images as input and predicts the labels. If we increase the size of A then accuracy increases and if we decrease its size then accuracy is also impacted.

### Varying B
**Prediction Trend Seen:** We don’t see any changes in the trend after changing this feature, we are not using this feature cause it’s not adding any value to our predication but is available in the original dataset.
No change in accuracy as we increase or decrease feature B.


### Varying A and B together
**Prediction Trend Seen:** If we vary them together then we can see that trend is changing and we are getting more accuracy than the previous. We can see that everything is dependent on feature A only and which is the main reason for our output accuracy. If we increase the data size the accuracy is increased as little.


### Varying A and B inversely
**Prediction Trend Seen:** If we change them inversely then we can see that accuracy is impacted cause we have reduced the data. The trend also reacts inversely as we reduce the data size then our accuracy is also reduced.