library(readxl)
library(tidyverse)
library(ggfortify)

library(mlbench)
library(e1071)
library(caret)


#LendingClub

clubData = read_excel('Lending.xlsx')


#Naive Bayes

columnsChange = c("loan_default", "loan_amnt", "adjusted_annual_inc",	"pct_loan_income", "dti",	"residence_property",	"months_since_first_credit", "open_acc", "bc_util", "pub_rec_bankruptcies")

#sapply allows you to loop through and applies the function to each column
sapply(clubData[columnsChange], unique)
#lapply allows you to loop through, in addition it returns
# a list or dataframe
clubData[columnsChange]  = lapply(clubData[columnsChange], as.factor)

## 75% of the sample size
smp_size <- floor(0.75 * nrow(clubData))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(clubData)), size = smp_size)

train <- clubData[train_ind, ]
test <- clubData[-train_ind, ]

NVmodel <- naiveBayes(loan_default ~ ., data = train)
preds <- predict(NVmodel, newdata = test)
conf_matrix <- table(preds, test$loan_default)

conf_matrix
confusionMatrix(conf_matrix)

## check the raw, this gives you the probability 
#predsRaw <- predict(NVmodel, newdata = test, type = "raw")
#predsRaw

library(ROCR)


# Compute AUC for predicting Class with the model
prob <- predict(NVmodel, newdata=test, type="raw")
pred <- prediction(prob[,2], test$loan_default)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
#following order: bottom, left, top, and right. 
par(mar=c(5,8,1,.5))
#Receiver operating characteristic
plot(perf, col="red")
abline(a=0, b=1)
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]
auc

rocData = data.frame(c(perf@alpha.values, perf@x.values, perf@y.values))


#logistic regression

data = read_excel('Lending.xlsx')

# The I() function creates a logical vector that is TRUE when loan_default is 1
# and FALSE otherwise
#lendingData = data %>% mutate(loan_default = I(loan_default == 1) %>% as.numeric())
#Separating Test and Training Data
data$loan_default = as.numeric(data$loan_default == 1)


columnsChange = c("loan_amnt", "adjusted_annual_inc",	"pct_loan_income",	"dti",	"residence_property",	"months_since_first_credit",	"inq_last_6mths",	"open_acc",	"bc_util",	"num_accts_ever_120_pd",	"pub_rec_bankruptcies")

#sapply allows you to loop through and applies the function to each column
#sapply(lendingData[columnsChange], unique)
#lapply allows you to loop through, in addition it returns
# a list or dataframe
#lendingData[columnsChange]  = lapply(lendingData[columnsChange], as.factor)
#glimpse(lendingData)
#TrainIndex = sample(1:nrow(lendingData), round(0.7*nrow(lendingData)))
TrainIndex = createDataPartition(data$loan_default, p = 0.7, list = FALSE)
#lendingTrain = lendingData[TrainIndex, ] 
#lendingTest = lendingData[-TrainIndex, ]
lendingTrain = data[TrainIndex, ] 
lendingTest = data[-TrainIndex, ]
#glimpse(lendingTrain)

lendLogit = glm(loan_default ~ ., data = lendingTrain[, c("loan_default", columnsChange)],# same as in lm()
                 family = "binomial") # for logistic, this is always set to "binomial"

summary(lendLogit)

varImp(lendLogit, scale=FALSE)
#creates a new column called EstimatedProb in HeartTest
EstimatedProb = predict(lendLogit,
                        newdata = lendingTest, type = "terms")

lendingTest = lendingTest %>% 
  mutate(EstimatedProb = predict(lendLogit,
                                 newdata = lendingTest, type = "response"))
summary(lendingTest$EstimatedProb)


lendTest2 = lendingTest %>% mutate(LendLogitPredicited = I(EstimatedProb > 0.5) %>% as.numeric())
#glimpse(HeartTest2)

heartTable = table(lendTest2$LendLogitPredicited ,lendTest2$loan_default)

confusionMatrix(heartTable)

library(pROC) 
library(PRROC)

# Compute AUC for predicting Class with the model
prob <- predict(lendLogit, newdata=lendingTest, type="response")
pred <- prediction(prob, lendingTest$loan_default)

#perf contains the data for drawing ROC curve 
perf <- performance(pred, measure = "tpr", x.measure = "fpr")

#following order: bottom, left, top, and right. 
par(mar=c(5,8,1,.5))
#Receiver operating characteristic
plot(perf, col="red")
abline(a=0, b=1)
auc <- performance(pred, measure = "auc")
auc@y.values[1]
auc