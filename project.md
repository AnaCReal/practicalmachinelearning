---
title: "Machine Learning peer-graded assignment"
author: "Ana Real"
date: "April 11, 2019"
output: 
  html_document:
    keep_md: true
---



## Prediction Assignment Writeup.

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants.

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

### Reading the data


```r
training <- read.csv2("pml-training.csv", header=TRUE, sep=",",na.strings = c("NA","DIV/0!",""))
testing <- read.csv2("pml-testing.csv", header=TRUE, sep=",",na.strings = c("NA","DIV/0!",""))
```

### Cleaning the data

Removing columns with many missing values and the columns that do not have to do with the exercise, in this case the first 7 variables from the dataset. I also had to convert to numeric variables as R was reading some of them as factors.


```r
cols = sapply(1:160,function(x)sum(is.na(training[,x])))
numcol = which(cols>0)
training <- training[,-numcol]
training <- training[,-c(1:7)]
for(i in 1:52){
    training[,i] <- as.numeric(as.character(training[,i]))
}
testing <- testing[,-numcol]
testing <- testing[,-c(1:7)]
for(i in 1:52){
    testing[,i] <- as.numeric(as.character(testing[,i]))
}
```

### Trainin the model

Since the goal variable "classe" is categorical using decision trees is a good model to compare random forest will be implemented with cross validation for 3. I also decided to create a validation set to have a better idea of the accuracy of the models.


```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
set.seed(101)
inTrain <- createDataPartition(training$classe, p=0.8, list=FALSE)
training <- training[inTrain,]
validation <- training[-inTrain,]
cross <- trainControl(method = "cv", number = 3)
```


```r
# Trees
model1 <- train(classe~., data=training, method = "rpart", trControl = cross)
```


```r
# Random Forest
model2 <- train(classe~., data=training, method = "rf", trControl = cross)
```

### Checking accuracy in the training set


```r
# For trees
pmodel1 <- predict(model1, training)
confusionMatrix(pmodel1, training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3051  969  918  866  302
##          B   58  566   37  338  118
##          C  232  744 1099  726  746
##          D    0    0    0    0    0
##          E    7    0    0    0  999
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4853          
##                  95% CI : (0.4762, 0.4944)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3281          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9113  0.24835  0.53505   0.0000  0.46143
## Specificity            0.6375  0.94198  0.74820   1.0000  0.99927
## Pos Pred Value         0.4997  0.50671  0.30984      NaN  0.99304
## Neg Pred Value         0.9476  0.83929  0.88395   0.8361  0.89174
## Prevalence             0.2843  0.19353  0.17442   0.1639  0.18385
## Detection Rate         0.2591  0.04806  0.09333   0.0000  0.08483
## Detection Prevalence   0.5185  0.09485  0.30121   0.0000  0.08543
## Balanced Accuracy      0.7744  0.59517  0.64163   0.5000  0.73035
```

```r
confusionMatrix(pmodel1, training$classe)$overall[1]
```

```
##  Accuracy 
## 0.4853091
```

```r
# For random forest
pmodel2 <- predict(model2, training)
confusionMatrix(pmodel2, training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3348    0    0    0    0
##          B    0 2279    0    0    0
##          C    0    0 2054    0    0
##          D    0    0    0 1930    0
##          E    0    0    0    0 2165
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

```r
confusionMatrix(pmodel2, training$classe)$overall[1]
```

```
## Accuracy 
##        1
```

### Evaluation in the validation set


```r
# For trees
pmodel1t <- predict(model1, validation)
confusionMatrix(pmodel1t, validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 811 247 233 228  82
##          B   9 156   4  93  30
##          C  73 210 288 211 202
##          D   0   0   0   0   0
##          E   1   0   0   0 254
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4818          
##                  95% CI : (0.4642, 0.4995)
##     No Information Rate : 0.2854          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3256          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9072  0.25449  0.54857   0.0000  0.44718
## Specificity            0.6470  0.94601  0.73303   1.0000  0.99961
## Pos Pred Value         0.5066  0.53425  0.29268      NaN  0.99608
## Neg Pred Value         0.9458  0.83908  0.88966   0.8301  0.89086
## Prevalence             0.2854  0.19572  0.16762   0.1699  0.18135
## Detection Rate         0.2589  0.04981  0.09195   0.0000  0.08110
## Detection Prevalence   0.5112  0.09323  0.31418   0.0000  0.08142
## Balanced Accuracy      0.7771  0.60025  0.64080   0.5000  0.72340
```

```r
confusionMatrix(pmodel1t, validation$classe)$overall[1]
```

```
##  Accuracy 
## 0.4818008
```

```r
# For random forest
pmodel2t <- predict(model2, validation)
confusionMatrix(pmodel2t, validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 894   1   0   0   0
##          B   0 611   2   0   0
##          C   0   1 522   3   0
##          D   0   0   1 529   0
##          E   0   0   0   0 568
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9974         
##                  95% CI : (0.995, 0.9989)
##     No Information Rate : 0.2854         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9968         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9967   0.9943   0.9944   1.0000
## Specificity            0.9996   0.9992   0.9985   0.9996   1.0000
## Pos Pred Value         0.9989   0.9967   0.9924   0.9981   1.0000
## Neg Pred Value         1.0000   0.9992   0.9988   0.9988   1.0000
## Prevalence             0.2854   0.1957   0.1676   0.1699   0.1814
## Detection Rate         0.2854   0.1951   0.1667   0.1689   0.1814
## Detection Prevalence   0.2858   0.1957   0.1679   0.1692   0.1814
## Balanced Accuracy      0.9998   0.9980   0.9964   0.9970   1.0000
```

```r
confusionMatrix(pmodel2t, validation$classe)$overall[1]
```

```
##  Accuracy 
## 0.9974457
```

### Getting predictions for test set


```r
# With trees
p1 <- predict(model1,testing)
p1
```

```
##  [1] C A C A A C C A A A C C C A C A A A A C
## Levels: A B C D E
```

```r
# With Random Forest
p2 <- predict(model2,testing)
p2
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
# Comparing
table(p1,p2)
```

```
##    p2
## p1  A B C D E
##   A 7 3 0 0 1
##   B 0 0 0 0 0
##   C 0 5 1 1 2
##   D 0 0 0 0 0
##   E 0 0 0 0 0
```

### Conclusions

Random forest got better accuracy in the training and validation for training it was perfect and for validation it was .99. In the prediction for the test set it is possible to see that tress could not predict labels D and E. So random forest was the best model for this data set.
