#R-codes for implementing four boosting algorithms (using the open-source R packages: caTools, gbm, xgboost, andadabag and configured for the lithofacies prediction using the reference well data.
1. Data Uploading and Visualization in R
setwd("~/Desktop/RWork")
data <- read.csv("Mishrif.csv",header=TRUE)
head(data)
par(mfrow=c(1,4))
plot(y=y<-(data$DEPTH),ylim=rev(range(data$DEPTH)),x=x<-(data$GR),
     type="l", col="red", lwd = 5, pch=17, xlab='Gamma Ray',
     ylab='Depth, m', xlim=c(0,90), cex=1.5, cex.lab=1.5, cex.axis=1.2)
plot(y=y<-(data$DEPTH),ylim=rev(range(data$DEPTH)),x=x<-(data$PHIT),
     type="l", col="blue", lwd = 5, pch=17, xlab='Log Porosity',
     ylab='Depth, m', xlim=c(0,0.25), cex=1.5, cex.lab=1.5, cex.axis=1.2)
plot(y=y<-(data$DEPTH),ylim=rev(range(data$DEPTH)),x=x<-(data$SW),
     type="l", col="darkgreen", lwd = 5, pch=17, xlab='Water Saturation',
     ylab='Depth, m', xlim=c(0,1), cex=1.5, cex.lab=1.5, cex.axis=1.2)
plot(y=y<-(data$DEPTH),ylim=rev(range(data$DEPTH)),x=x<-(data$Facies),
     type="l", col="gold", lwd = 5, pch=17, xlab='Facies',
     ylab='Depth, m', cex=1.5, cex.lab=1.5, cex.axis=1.2)

2. Cross-Validation Process
data <- read.csv("Mishrif.csv",header=TRUE)
n = nrow(data)
train.index = sample(n,floor(0.75*n))
train = data[train.index,]
test = data[-train.index,]
3. R Codes
a. Logistic Boosting Regression (LogitBoost)
require(caTools)
library(caTools)
#Dataset needs to be split into continuous and discrete parameters.
xlearn  = train[c(1,2,3,4,5,6,7)]
ylearn = train[, 8]
# nIter: An integer describes the number of iterations for-
# -which boosting should be run.
# nIter=ncol(xlearn)
model = LogitBoost(xlearn, ylearn, nIter=7)
#Prediction for the entire dataset
Lab = predict(model, majn)
#Modeling Validation by computing the total correct percent.
Labelm <- data[,8]
ct1 <- table(predict(model, data), Labelm)
# The total percent correct:
diag(prop.table(ct1))
sum(diag(prop.table(ct1)))
par(mfrow=c(1,2))
plot(y=y<-(majn$DEPTH),ylim=rev(range(majn$DEPTH)),x=x<-(majn$Facies),
     type="l", col="red", lwd = 5, pch=17, xlab='Measured Facies', notch=TRUE,
     ylab='Depth, m', cex=1.5, cex.lab=1.5, cex.axis=1.2)
plot(y=y<-(majn$DEPTH),ylim=rev(range(majn$DEPTH)),x=x<-(Lab), type="l", 
     col="darkgreen", lwd = 5, pch=17, xlab='Predicted Facies', notch=TRUE,
     ylab='Depth, m', cex=1.5, cex.lab=1.5, cex.axis=1.2, main="TPC=0.9765")
#Prediction for the test dataset
Labt = predict(model, test)
#Modeling Validation by computing the total correct percent.
Labelt <- test[,8]
ct2 <- table(predict(model, test), Labelt)
# The total percent correct:
diag(prop.table(ct2))
sum(diag(prop.table(ct2)))
par(mfrow=c(1,2))
plot(y=y<-(test$DEPTH),ylim=rev(range(test$DEPTH)),x=x<-(test$Facies),
     type="l", col="red", lwd = 5, pch=17, xlab='Measured Facies', notch=TRUE,
     ylab='Depth, m', cex=1.5, cex.lab=1.5, cex.axis=1.2)
plot(y=y<-(test$DEPTH),ylim=rev(range(test$DEPTH)),x=x<-(Labt), type="l", 
     col="green", lwd = 5, pch=17, xlab='Predicted Facies', notch=TRUE, 
     ylab='Depth, m', cex=1.5, cex.lab=1.5, cex.axis=1.2,
     main="TPC=0.94")
b. Generalized Boosting Modeling (GBM)
require(gbm)
require(caret)
library(gbm)
library(caret)
gbm <- gbm(Facies ~ ., 
           data              = train, 
           distribution = "multinomial",
           cv.folds = 10,
           shrinkage = .01,
           n.minobsinnode = 10,
           n.trees = 200)
# The default settings in gbm includes a shrinkage (learning rate) of 0.001.
This is a very small learning rate and typically requires a large number of
trees to find the minimum MSE. However, gbm uses a default number of trees
of 100, which is rarely sufficient. Consequently, in the models used in this
study shrinkage was set at 0.01 with 200 trees. A 10-fold cross validation
technique is applied to minimize the MSE loss function with 200 trees.
#Prediction for the entire dataset
pred.gbm = predict.gbm(object = gbm,
                       newdata = majn,
                       n.trees = 200,
                       type = "response")
# Type of predicted outcome is "response" to have class (discrete) distribution
of facies; NOT posterior distribution.
labels1 = colnames(pred.gbm)[apply(pred.gbm, 1, which.max)]
cm = confusionMatrix(majn$Facies, as.factor(labels1))
ct3 <- table(predict(gbm, majn[,-8]))
# The total percent correct:
diag(prop.table(ct3))
sum(diag(prop.table(ct3)))
#Prediction for the test subset
pred.gbm = predict.gbm(object = gbm,
                       newdata = test,
                       n.trees = 200,
                       type = "response")
labels2 = colnames(pred.gbm)[apply(pred.gbm, 1, which.max)]
cm = confusionMatrix(test$Facies, as.factor(labels2))
result = data.frame(test$Facies, labels2)
ct4 <- table(predict(gbm, test[,-8]))
# The total percent correct:
diag(prop.table(ct4))
sum(diag(prop.table(ct4)))
c. Extreme Gradient Boosting (XGBoost)
require(xgboost)
library(xgboost)
label = as.integer(data$Facies)-1
# Transform the two data sets into xgb.Matrix
xgb = xgb.DMatrix(data=as.matrix(data[,-8]),label=label)
# Define the parameters for multinomial classification
num_class = length(levels(lithofacies))
params = list(        # tuning of the XGBoost model
  booster="gbtree",   # tree based models (gbtree) or linear functions(gblinear).
  eta=0.001,          # the learning rate represents step size shrinkage
                        to prevents overfitting
  max_depth=5,        # maximum depth of a tree. Increasing it makes the
                        model more complex and more likely to overfit.
  gamma=3,            # minimum loss reduction required to make a further
                        partition on a leaf node of the tree.
  subsample=0.75,     # percent of training data to sample for each tree.
                        range: (0,1].
  colsample_bytree=1, # percent of columns to sample from for each tree.
                        [default=1]. 
  objective="multi:softprob",  # multiclass classification
  eval_metric="mlogloss",      # evaluation metric of multi-class log loss 
  num_class=num_class          # number of classes
)
# The values of the tuning parameters to control overfitting in XGBoost have
 been set to the default values.
# Some of these parameters works to directly control model complexity such as
 max_depth and gamma. Range: $[0,\infty]$
# Other terms act to add randomness to make training robust to noise such as
 subsample and colsample_bytree. 
# To train the XGBoost classifer:
xgb.fit=xgb.train(
  params=params,
  data=xgb,
  nrounds=10000,              # maximum number of interations
  nthreads=1,                 # number of parallel threads used to run XGBoost
  early_stopping_rounds=10,   # stopping criteria
  watchlist=list(val1=xgb,val2=xgb),
  verbose=0
)
# XGBoost training procedure supports early stopping creteria after a fixed number 
of iterations to stop model training before the model overfits the training data. 
# In the early_stopping_rounds parameter, users must specify a window of the
 number of epochs over which no improvement is observed.
# For example, we can check for no improvement in loss function over 10
 epochs (early_stopping_rounds=10).
# Predict outcomes with the entire data
xgb.pred = predict(xgb.fit,as.matrix(majn[,-8]),reshape=T)
xgb.pred = as.data.frame(xgb.pred)
colnames(xgb.pred) = levels(lithofacies)
# Use the predicted label with the highest probability
xgb.pred$prediction=apply(xgb.pred,1,function(x)colnames(xgb.pred)[which.max(x)])
xgb.pred$label = levels(lithofacies)[label+1]
# compute the total correct percent.
ct5 <- table(xgb.pred$prediction, label)
diag(prop.table(ct5))
# The total percent correct:
sum(diag(prop.table(ct5)))
PredL <- as.data.frame(xgb.pred$prediction, quote=FALSE)
# Predict outcomes with the test data
xgb.predt = predict(xgb.fit,as.matrix(test[,-8]),reshape=T)
xgb.predt = as.data.frame(xgb.predt)
colnames(xgb.predt) = levels(lithofaciest)
# Use the predicted label with the highest probability
xgb.predt$prediction=apply(xgb.predt,1,function(x)colnames(xgb.predt)[which.max(x)])
xgb.predt$labelt = levels(lithofaciest)[labelt+1]
# compute the total correct percent.
ct6 <- table(xgb.predt$prediction, labelt)
diag(prop.table(ct6))
# The total percent correct:
sum(diag(prop.table(ct6)))
PredLt <- as.data.frame(xgb.predt$prediction, quote=FALSE)
d. Adaptive Boosting Model (AdaBoost)
require(adabag)
library(adabag)
model = boosting(Facies~., data=train, boos=TRUE, mfinal=100)
# mfinal: the number of iterations to run the boosting model.
  Defaults to mfinal=100 iterations.
pred = predict(model, data)
result = data.frame(majn$Facies, pred$prob, pred$class)
### Testing Subset
predt = predict(model, test)
print(predt$confusion)
print(predt$error)
resultt = data.frame(test$Facies, predt$prob, predt$class)