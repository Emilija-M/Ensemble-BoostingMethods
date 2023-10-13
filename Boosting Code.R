###Ensemble - Boosting Methods

#Classification

library(caret)

#For training/test separation
install.packages("gbm")
library(gbm)

#For gradient Boosting
install.packages("adabag")
library(adabag)

#For AdaBoost

#We are loading the wdbc database from the mclust package, which tells us about data for breast cancer in women
#Diagnosis column represents a dependent variable with two levels, M - which means malignant
#and B - which means benign
library(mclust)
library(caret)
data <- wdbc
head(data)

#We divide the data into training and test set in a ratio of 70:30
indeks <- createDataPartition(data$Diagnosis, p=0.7, list=FALSE)
trening <- data[indeks,]
test <- data[-indeks,]

#We build a gbm model with 1000 trees

gbm_model = gbm(Diagnosis ~.,
                data = trening,
                distribution = "multinomial",
                cv.folds = 10,
                shrinkage = 0.01,
                n.minobsinnode = 10,
                n.trees = 1000)       

summary(gbm_model)
#We see that the most significant predictors are: Nconcave_extreme, Perimeter_extreme, Nconcave_mean i Radius_Extreme

#We make predictions on the test set
gbm_pred = predict.gbm(object = gbm_model,
                       newdata = test,
                       n.trees = 1000,           
                       type = "response")

gbm_pred
class_names = colnames(gbm_pred)[apply(gbm_pred, 1, which.max)]
result = data.frame(test$Diagnosis, class_names)
result

#Confusion matrix
conf_mat = confusionMatrix(test$Diagnosis, as.factor(class_names))
conf_mat$table
conf_mat$overall[1]

#Accuracy 97% 
#Error 3% 
#105 2    
#3 60     

ada_model <- boosting(Diagnosis~ ., data = trening, mfinal=1000)
ada_model$importance
#We do not get the same predictors as the most significant, we see that here both Texture_extreme and Texture_mean are significant
plot(ada_model$importance, type = "h")

ada_pred <- predict(ada_model, test)
ada_pred
1 - ada_pred$error
ada_pred$confusion
#Error 5.3%
#Accuracy 94.7%
#103 5
#4 58

#We create a logistic model with significant predictors
log_model <- glm(Diagnosis ~ Nconcave_extreme + Perimeter_extreme + Nconcave_mean + Radius_extreme, trening, family = binomial(link = "logit"))
summary(log_model)
log_model_test <- glm(Diagnosis ~ Nconcave_extreme + Perimeter_extreme + Nconcave_mean + Radius_extreme, test, family = binomial(link = "logit"))

log_pred <- predict(log_model_test, type="response") > 0.5
log_pred
confusion <- table(test$Diagnosis, log_pred)
confusion
library(Metrics)
accuracy(test$Diagnosis == "M", log_pred)
#94.1%

#Now we are building a model with all predictors, we expect better precision
log_model_sve <- glm(Diagnosis ~ ., trening, family = binomial(link = "logit"))
summary(log_model_sve)
log_model_test_sve <- glm(Diagnosis ~ ., test, family = binomial(link = "logit"))

log_pred_sve <- predict(log_model_test_sve, type="response") > 0.5
log_pred_sve
confusion_sve <- table(test$Diagnosis, log_pred_sve)
confusion_sve
accuracy(test$Diagnosis == "M", log_pred_sve)
#When we use all predictors, we get accuracy 1
#However, R showed us a warning that the model did not converge,
#Which means that with logistic regression on the training and test set, we perfectly divided the data
#This could mean that we are in overfit
#So we will use the results obtained with significant predictors

tabela <- data.frame(model = c("Gradient Boosting", "Adaboost", "Logisticki"),
                     accuracy = c(conf_mat$overall[1]*100 , (1 - ada_pred$error)*100, accuracy(test$Diagnosis == "M", log_pred)*100))
tabela
#Conclusion: We saw that the best model was GBM, followed by AdaBoost, and finally logistic regression.

###Regression

library(MASS)
summary(Boston)

### GBM

#We use the Boston database, we want to determine the relationship between the house price and other predictors.
#We divide the data into training and test set in a ratio of 70:30.
indeks <- createDataPartition(Boston$medv, p=0.7, list=FALSE)
trening <- Boston[indeks,]
test <- Boston[-indeks,]

#We build a GBM model with 1000 trees, we use a Gaussian normal distribution. 
model_gbm = gbm(trening$medv ~ .,
                data = trening,
                distribution = "gaussian",
                cv.folds = 10,
                shrinkage = .01,
                n.minobsinnode = 10,
                n.trees = 1000)

summary(model_gbm)
#We get that lstat and rm are the most significant predictors, with a smaller influence of the others.

#We extract the dependent variable and predictors on the test set.
test_X <- test[,-14]
test_y <- test[,14]

#We make predictions on the test set.
pred_y = predict.gbm(model_gbm, test_X)
pred_y

#We calculate the mean squared error and then the R-squared to evaluate the model.
res = test_y - pred_y
RMSE = sqrt(mean(res^2))
RMSE
#RMSE: 3.28

y_test_mean = mean(test_y)
tss =  sum((test_y - y_test_mean)^2 )
rss =  sum(res^2)
rsq  =  1 - (rss/tss)
rsq
#R-squared: 0.83

#We plot real and predicted data.
x_ax = 1:length(pred_y)
plot(x_ax, test_y, col="blue", pch=20, cex=.9)
lines(x_ax, pred_y, col="red", pch=20, cex=.9) 
#The model evaluates the data well.

###Linear model

lin_model <- lm(medv ~ ., trening)
summary(lin_model)
#R-squared: 0.73
#Adjusted R-squared: 0.72
lin_pred <- predict(lin_model, test)
lin_pred

#We conclude that the GBM regression model performed better than the linear model with the same predictors.
#We will now try to transform the data and perform some analyses, in order to improve the linear model.

#We will examine the conditions:

##Symmetry around zero:
library(car)
residualPlots(lin_model)
#With the predictor rm there is a quadratic dependence, while with lstat it acts as an exponential one

#Testing the normality of the residuals:
qqPlot(rstandard(lin_model))
#They fall out of the blue area, we cannot say that they are normally distributed
#We will also check with the Shapiro test
shapiro.test(rstandard(lin_model))
#Small p-value, we reject normality

#Homoscedasticity:
plot(abs(rstandard(lin_model)) ~ fitted(lin_model))
#Heteroskedasticity is present

#To get a better model, we will solve these problems
#We will throw out the age and indus predictors because they are not significant

sub_model <- lm(medv ~ . - indus - age, trening)
summary(sub_model)
#With this we only increased the Adjusted R-squared

#Now we will transform the predictors lstat and rm
lin_model_2 <- lm(medv ~ . - age - indus + I(log(lstat)) + I(rm^2), trening)
summary(lin_model_2)
#We increased R-squared to 0.81, which is still less than what we got for GBM

#We will check the residuals again
residualPlots(lin_model_2)
#Residuals look better now
qqPlot(rstandard(lin_model_2))
plot(abs(rstandard(lin_model_2)) ~ fitted(lin_model_2))

#To solve the heteroscedasticity, we will apply the Box-Cox transformation
bc <- boxcox(lin_model_2)
bc$x[which.max(bc$y)]
# In this case lambda is around 0.30
BC <- function(y) (y^0.30 - 1)/0.30
plot(BC(medv) ~ . - age - indus + I(log(lstat)) + I(rm^2), trening)
lin_model_bc <- lm(BC(medv) ~ . - age - indus + I(log(lstat)) + I(rm^2), trening)
summary(lin_model_bc)
qqPlot(rstandard(lin_model_bc))
#We also fixed qqplot, it looks more acceptable
#We got an R-squared of 0.826

#We conclude that even with predictor transformations and data correction, we get a worse model than GBM
#With dropping and transforming predictors, as well as Box-Cox transformation, we approximated GBM to less than 1 difference in R-squared.

#GBM is a better regression model if we do not want to transform and drop predictors.
