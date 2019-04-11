---
title: "Practical Machine Learning Project"
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
training <- read.csv2("./../pml-training.csv", header=TRUE, sep=",",na.strings = c("NA","DIV/0!",""))
testing <- read.csv2("./../pml-testing.csv", header=TRUE, sep=",",na.strings = c("NA","DIV/0!",""))
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
##          A 4057 1245 1264 1147  416
##          B   68 1026   83  444  373
##          C  327  767 1391  982  777
##          D    0    0    0    0    0
##          E   12    0    0    0 1320
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4965          
##                  95% CI : (0.4886, 0.5043)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3422          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9088  0.33772   0.5080   0.0000  0.45738
## Specificity            0.6376  0.92354   0.7799   1.0000  0.99906
## Pos Pred Value         0.4991  0.51454   0.3278      NaN  0.99099
## Neg Pred Value         0.9462  0.85319   0.8824   0.8361  0.89100
## Prevalence             0.2843  0.19352   0.1744   0.1639  0.18383
## Detection Rate         0.2584  0.06535   0.0886   0.0000  0.08408
## Detection Prevalence   0.5178  0.12701   0.2703   0.0000  0.08485
## Balanced Accuracy      0.7732  0.63063   0.6440   0.5000  0.72822
```

```r
confusionMatrix(pmodel1, training$classe)$overall[1]
```

```
##  Accuracy 
## 0.4964647
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
##          A 4464    0    0    0    0
##          B    0 3038    0    0    0
##          C    0    0 2738    0    0
##          D    0    0    0 2573    0
##          E    0    0    0    0 2886
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9998, 1)
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
##          A 808 244 232 225  82
##          B   9 214  12  97  71
##          C  76 155 281 210 161
##          D   0   0   0   0   0
##          E   1   0   0   0 254
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4971          
##                  95% CI : (0.4795, 0.5148)
##     No Information Rate : 0.2854          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.345           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9038  0.34910  0.53524   0.0000  0.44718
## Specificity            0.6501  0.92497  0.76908   1.0000  0.99961
## Pos Pred Value         0.5079  0.53102  0.31823      NaN  0.99608
## Neg Pred Value         0.9442  0.85379  0.89151   0.8301  0.89086
## Prevalence             0.2854  0.19572  0.16762   0.1699  0.18135
## Detection Rate         0.2580  0.06833  0.08972   0.0000  0.08110
## Detection Prevalence   0.5080  0.12867  0.28193   0.0000  0.08142
## Balanced Accuracy      0.7770  0.63704  0.65216   0.5000  0.72340
```

```r
confusionMatrix(pmodel1t, validation$classe)$overall[1]
```

```
##  Accuracy 
## 0.4971264
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
##          A 894   0   0   0   0
##          B   0 613   0   0   0
##          C   0   0 525   0   0
##          D   0   0   0 532   0
##          E   0   0   0   0 568
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9988, 1)
##     No Information Rate : 0.2854     
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
## Prevalence             0.2854   0.1957   0.1676   0.1699   0.1814
## Detection Rate         0.2854   0.1957   0.1676   0.1699   0.1814
## Detection Prevalence   0.2854   0.1957   0.1676   0.1699   0.1814
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

```r
confusionMatrix(pmodel2t, validation$classe)$overall[1]
```

```
## Accuracy 
##        1
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
