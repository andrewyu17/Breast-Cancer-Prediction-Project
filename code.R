## ML Breast Cancer Project

## Packages/Settings
options(digits = 3)
library(matrixStats)
library(tidyverse)
library(caret)
library(dslabs)
data(brca)

brca$x

### Dimensions and Properties

dim(brca$x) # 569 Samples with 30 Predictors

mean(brca$y == 'M') # 37.3 percent of the samples are malignant

which.max(colMeans(brca$x)) # Predictor 24 has the highest mean

which.min(colSds(brca$x)) # Predictor 20 has the lowest standard deviation

### Scaling the Matrix

x <- sweep(brca$x, 2, colMeans(brca$x))
scale <- sweep(x, 2, colSds(brca$x), FUN = '/')

sd(scale[,1]) # First predictor has standard deviation 1 after scaling

median(scale[,1]) # First predictor median value is -0.215 after scaling

### Distance

distance <- dist(scale)
sample1B <- as.matrix(distance)[1, brca$y == 'B']
mean(sample1B[2:length(sample1B)])
# Average distance between first benign sample and other benign samples is 4.41

sample1M <- as.matrix(distance)[1, brca$y == 'M']
mean(sample1M)
# Average distance between first malignant sample and other malignant samples is 7.12

### Heatmap of features
features <- dist(t(scale))
heatmap(as.matrix(features), labRow = NA, labCol = NA)


### Hierarchal Clustering
hc <- hclust(features)
groups <- cutree(hc, k = 5)
split(names(groups), groups)

### PCA Analysis
pca <- prcomp(scale)
summary(pca)
# First feature explains 44.3% of variance
# At least 7 features are required to explain 90% variance

### PCA Visualization
data.frame(pca$x[,1:2], type = brca$y) %>%
  ggplot(aes(PC1, PC2, color = type)) +
  geom_point()
# Malignant tumors tend to have larger values of PC1 than benign tumors

### PCA Boxplot
data.frame(pca$x[,1:10], type = brca$y) %>%
  gather(key = "PC", value = "value", -type) %>%
  ggplot(aes(PC, value, fill = type)) +
  geom_boxplot()
# Only PC1 is significant different by tumor type

### Training and Test Sets
set.seed(1)
test_index <- createDataPartition(brca$y, times = 1, p = 0.2, list = FALSE)
test_x <- scale[test_index,]
test_y <- brca$y[test_index]
train_x <- scale[-test_index,]
train_y <- brca$y[-test_index]

mean(train_y=='B') # 62.8% of the training set is benign
mean(test_y=='B') # 62.6% of the test set is bening

### K-means Clustering
predict_kmeans <- function(x, k) {
  centers <- k$centers    # extract cluster centers
  # calculate distance to cluster centers
  distances <- sapply(1:nrow(x), function(i){
    apply(centers, 1, function(y) dist(rbind(x[i,], y)))
  })
  max.col(-t(distances))  # select cluster with min distance to center
}

set.seed(3) 
k <- kmeans(train_x, centers = 2)
kmeans <- ifelse(predict_kmeans(test_x, k) == 1, "B", "M")
mean(kmeans_preds == test_y) # Accuracy of 92.2% with K-means clustering

### K-means Clustering Sensitivity
table(kmeans_preds, test_y)
sensitivity(factor(kmeans_preds), test_y, positive = "B") # 98.6% were correctly identified
sensitivity(factor(kmeans_preds), test_y, positive = "M") # 81.4% were correctly identified
# Our model is more accurate when identifying benign tumors

### Log Regression Model
set.seed(1)
glm <- train(train_x, train_y, method = 'glm')
glmpredicts <- predict(glm, test_x)
mean(glmpredicts == test_y) # Log Regression model has an accuracy of 95.7%

### LDA and QDA Models
set.seed(1)
lda <- train(train_x, train_y, method = 'lda')
ldapredicts <- predict(lda, test_x)
mean(ldapredicts == test_y) # LDA model has 99.1% accuracy

set.seed(1)
qda <- train(train_x, train_y, method = 'qda')
qdapredicts <- predict(qda, test_x)
mean(qdapredicts == test_y) # QDA model has 95.7% accuracy

### Loess Model
set.seed(5)
loessm <- train(train_x, train_y, method = 'gamLoess')
loessmpredicts <- predict(loessm, test_x)
mean(loessmpredicts == test_y) # Loess model has 98.3% accuracy

### KNN Model
set.seed(7)
knn <- train(train_x, train_y, method = 'knn', tuneGrid = data.frame(k = seq(3,21,2)))
knnpredicts <- predict(knn, test_x)
knn$bestTune # Model shows best value of k is 21 out of our range
mean(knnpredicts == test_y) # KNN model has 94.8% accuracy

### Random Forest Model
set.seed(9)
rf <- train(train_x, train_y, method = 'rf', 
            tuneGrid = data.frame(mtry = c(3, 5, 7, 9)),
            importance = TRUE)
rfpredicts <- predict(rf, test_x)
rf$bestTune # Best value of mtry is 3 from our range of values
mean(rfpredicts == test_y) # RF model has 94.8% accuracy
varImp(rf) # area_worst variable has highest importance in model

### Create Ensemble
ensemble <- cbind(glm = glmpredicts == "B", 
                  lda = ldapredicts == "B", 
                  qda = qdapredicts == "B", 
                  loess = loessmpredicts == "B", 
                  rf = rfpredicts == "B", 
                  knn = knnpredicts == "B", 
                  kmeans = kmeans == "B")
predicts <- ifelse(rowMeans(ensemble) > 0.5, 'B', 'M')
mean(predicts == test_y) # Ensemble model has 98.3% accuracy

### Which model in Ensemble has highest accuracy
models <- c('kmeans', 'glmpredicts', 'ldapredicts', 'qdapredicts', 
            'loessmpredicts', 'knnpredicts', 'rfpredicts')
pred <- as.data.frame(sapply(1:7, function(x){
  as.factor(get(models[x]))}))
names(pred) <- models
acc <- colMeans(as.matrix(pred)==test_y)
acc # LDA model has the highest accuracy

