#http://ceholden.github.io/open-geo-tutorial/R/chapter_5_classification.html

rasterOptions()


setwd("D:/ECSA5/") #set directory
getwd()
list.files()

#######################################################################################

library(raster)
library(rgdal)
library(sf)
library(reshape2)

library(caret)
library(lattice, ggplot2)
library(dplyr)

library(neuralnet)
library(nnet)
library(e1071)

library(doParallel)  #Foreach Parallel Adaptor 
library(foreach)     #Provides foreach looping construct

#Define how many cores you want to use
UseCores <- detectCores() -2

#Register CoreCluster
cl       <- makeCluster(UseCores)
registerDoParallel(cl)

#########################################################################################

#read tif file from landsat
ls8 <- brick('Full.tif')
ls81<- stack('Full.tif')
names(ls8)

#rename Landsat bands
ls8_class <- ls8
names(ls8_class)
names(ls8_class) <- c('b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7')
names(ls8_class)

#plots indiviual layers
#plot(ls8_class)

#plot RGB
plotRGB(ls8_class, r=4, g=3, b=2, stretch="lin")
#plotRGB(ls8_class, r=5, g=3, b=2, stretch="lin")

#########################################################################################
#https://www.neonscience.org/dc-shapefile-attributes-r

#read and prep shp file as training data
training1 <- read_sf(dsn='Random_pointsonly.shp', layer='Random_pointsonly')


# view class
training2 <- readOGR("Random_pointsonly.shp")
class(x = training2)

# view features count
length(training2)

# view crs - note - this only works with the raster package loaded
crs(training2)

# view extent- note - this only works with the raster package loaded
extent(training2)

# view metadata summary
training2

#########################################################################################
# Create color map

#Simple Colors
#ColorPalette <- c("blue","red")
#ColorPalette

#Mixed Colors
colors <- c(rgb(211, 211, 211, maxColorValue=255),  # Urban
            rgb(0, 0, 255, maxColorValue=255))      # Water

colors1 <- c(rgb(0, 0, 255, maxColorValue=255),          # Urban
             rgb(211, 211, 211, maxColorValue=255))      # Water

colors
#########################################################################################

#simply plot training data
#plot(training1)

#plot points only
#plot(training1, pch=16, col="green", add=TRUE)

#plot types of class
plot(training1, pch=16, col=colors, main="Training and validation points", add=TRUE)

#plot shp and legends
class(training2$class)
levels(training2$class)
length(levels(training2$class))

colors
WaterColors <- colors[training2$class]
WaterColors
plotRGB(ls8_class, r=5, g=3, b=2, stretch="lin")
plot(training2, pch=16, col=WaterColors, main="Training and validation points", add=TRUE)
#legend("bottomright", legend=levels(training2$class), fill=colors)
legend("bottomright", legend=c('Nonwater','Water'), fill=colors)

#########################################################################################

roi_data1 <- extract(ls8_class, training1, df=TRUE)
roi_data2 <- roi_data1

#Remember that this dataset may have clouds or cloud shadows. Let's mask them out:
#roi_data[which(roi_data$Band.8 > 1)] <- NA


# Set seed value for RBG for reproducibility
set.seed(123)
colnames(roi_data1)

# Shorten column names
#colnames(roi_data1) <- c('ID', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'lc', 'desc')
#colnames(roi_data1)

#We'll also need to attach the labels to this DataFrame:
roi_data1$lc <- as.factor(training1$ID)
#roi_data1$desc <- as.factor(training1$class)

#########################################################################################
#MUST SPLIT TRAIN and TEST later
#https://www.analyticsvidhya.com/blog/2017/09/creating-visualizing-neural-network-in-r/

#########################################################################################
#Unsupervised clustering
library("cluster")
library("factoextra")

roi_data3 <- roi_data2[-1]

res.dist <- get_dist(roi_data3, stand = TRUE, method = "pearson")
fviz_dist(res.dist, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))

#Determine the optimal number of clusters
fviz_nbclust(roi_data3, kmeans, method = "gap_stat")

#Compute and visualize k-means clustering
km.res <- kmeans(roi_data3, 2, nstart = 25)
# Visualize
fviz_cluster(km.res, data = roi_data3, frame.type = "convex")+
  theme_minimal()

# Compute PAM
pam.res <- pam(roi_data3, 2)

# Visualize
fviz_cluster(pam.res)

#Compute Hirarchial clustering
# 1. Loading and preparing data
#data("roi_data3")
#roi_data3 <- scale(roi_data3)
# 2. Compute dissimilarity matrix
d <- dist(roi_data3, method = "euclidean")
# Hierarchical clustering using Ward's method
res.hc <- hclust(d, method = "ward.D2" )
# Cut tree into 4 groups
grp <- cutree(res.hc, k = 2)
# Visualize
plot(res.hc, cex = 0.6) # plot tree
rect.hclust(res.hc, k = 2, border = 2:5) # add rectangle

# Visualize the silhouette plot
#fviz_silhouette(res.hc)


# Compute hierarchical clustering and cut into 4 clusters
res <- hcut(roi_data3, k = 4, stand = TRUE)

# Visualize
fviz_dend(res, rect = TRUE, cex = 0.5,k_colors = c("#00AFBB","#2E9FDF", "#E7B800", "#FC4E07"))


# Compute clValid and select Clustering method
library("clValid")
library("kohonen")
intern <- clValid(roi_data3[1:73,], nClust = 2:6, clMethods = c("hierarchical", "kmeans", "diana", "fanny", "sota", "pam", "clara", "agnes"), validation = "internal")
# Summary
summary(intern)


#########################################################################################
#https://gis.stackexchange.com/questions/123639/unsupervised-classification-with-kmeans-in-r

#Unsupervised classification not good for 2 class

###classify raster
#library(RStoolbox)
#set.seed(25)
#unC <- unsuperClass(ls8_class, nSamples = 25, nClasses = 2, nStarts = 5)
#unC
#plot(unC$map, col = colors, legend = FALSE, axes = FALSE, box = FALSE)
#legend(1,1, legend = paste0("C",1:2), fill = colors, title = "Classes", horiz = TRUE,  bty = "n")
#########################################################################################

#RANDOM FOREST
library(randomForest)
rf1 <- randomForest(lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data=roi_data1, importance=TRUE)
summary(rf1)

#stats
print(rf1)
plot(rf1)
varImpPlot(rf1)

# Predict!
ls8_pred_rf1 <- predict(ls8_class, model=rf1, na.rm=T)
plot(ls8_pred_rf1, col=colors)

#Confusion Matrix
summary(roi_data1[,9])
rf1_prediction <- predict(rf1, roi_data1[2:8])
summary(rf1_prediction)
table(rf1_prediction,roi_data1[,9])

cmresult <- confusionMatrix(table(rf1_prediction,roi_data1[,9]))
cmresult
sink("LS8_rf1_CM.txt"); cmresult; sink()


#tuneRF

#########################################################################################

#SVM
library("e1071")
svm1 <- svm(lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data=roi_data1)
summary(svm1)

#prediction
ls8_pred_svm1 <- predict(ls8_class, model=svm1, na.rm=T)
#plot(ls8_pred_svm1, col=colors)

#Confusion Matrix
summary(roi_data1[,9])
svm1_prediction <- predict(svm1, roi_data1[2:8])
summary(svm1_prediction)
table(svm1_prediction,roi_data1[,9])


cmresult <- confusionMatrix(table(svm1_prediction,roi_data1[,9]))
cmresult
sink("LS8_svm1_CM.txt"); cmresult; sink()

#SVM_tuning
svm_tune1 <- tune.svm(lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data=roi_data1, cost = 10^(-2:2), gamma = 10^(-2:2))
summary(svm_tune1)
plot(svm_tune1)

#prediction
svm_model_after_tune1 <- svm(lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data=roi_data1, kernel="radial", cost=1, gamma=0.01)
ls8_pred_svm_after_tune1 <- predict(ls8_class, model=svm_model_after_tune1, na.rm=T)
#plot(ls8_pred_svm_after_tune1, col=colors)

#Confusion Matrix
summary(roi_data1[,9])
svm_tune1_prediction <- predict(svm_model_after_tune1, roi_data1[2:8])
summary(svm_tune1_prediction)
table(svm_tune1_prediction,roi_data1[,9])

cmresult <- confusionMatrix(table(svm_tune1_prediction,roi_data1[,9]))
cmresult
sink("LS8_svm_tune1_CM.txt"); cmresult; sink()

#SVM_tuning
svm_tune2 <- tune(svm, lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data=roi_data1, kernel="radial", ranges=list(cost=10^(-2:2), gamma=c(.5,1,2,4)))
summary(svm_tune2)
plot(svm_tune2)

#prediction
svm_model_after_tune2 <- svm(lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data=roi_data1, kernel="radial", cost=1, gamma=0.5)
ls8_pred_svm_after_tune2 <- predict(ls8_class, model=svm_model_after_tune2, na.rm=T)
#plot(ls8_pred_svm_after_tune2, col=colors)

#Confusion Matrix
summary(roi_data1[,9])
svm_tune2_prediction <- predict(svm_model_after_tune2, roi_data1[2:8])
summary(svm_tune2_prediction)
table(svm_tune2_prediction,roi_data1[,9])

cmresult <- confusionMatrix(table(svm_tune2_prediction,roi_data1[,9]))
cmresult
sink("LS8_svm_tune2_CM.txt"); cmresult; sink()

#stats


#########################################################################################
#Recursive Partitioning and Regression Trees
library(rpart)
rp1 <- rpart(lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data=roi_data1)
summary(rp1)

#tree
print(rp1)
plot(rp1)
text(rp1)

#Stats
printcp(rp1)
plotcp(rp1)

rsq.rpart(rp1) # cross-validation results 

# Predict!
ls8_pred_rp1 <- predict(ls8_class, model=rp1, na.rm=T)
plot(ls8_pred_rp1, col=colors1)

#test error due to 0 1 probability so add type=class
summary(roi_data1[,9])
rp1_prediction <- predict(rp1, roi_data1[2:8], type="class")
summary(rp1_prediction)
table(rp1_prediction,roi_data1[,9])

cmresult <- confusionMatrix(table(rp1_prediction,roi_data1[,9]))
cmresult
sink("LS8_rp1_CM.txt"); cmresult; sink()






#rp_tuning
rp_tune1 <- tune.rpart(lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data=roi_data1, minsplit = c(2, 3, 4,5,10,15, 20))
summary(rp_tune1)
plot(rp_tune1)



#tree
print(rp_tune1)
plot(rp1)
text(rp1)

#Stats
printcp(rp1)
plotcp(rp1)

rsq.rpart(rp1) # cross-validation results 

# Predict!
ls8_pred_rp1 <- predict(ls8_class, model=rp1, na.rm=T)
plot(ls8_pred_rp1, col=colors1)

#test error due to 0 1 probability so add type=class
summary(roi_data1[,9])
rp1_prediction <- predict(rp1, roi_data1[2:8], type="class")
summary(rp1_prediction)
table(rp1_prediction,roi_data1[,9])

cmresult <- confusionMatrix(table(rp1_prediction,roi_data1[,9]))
cmresult
sink("LS8_rp1_CM.txt"); cmresult; sink()



#######################################################################################

#ann
#NNNET WORKS
library(nnet)
library(NeuralNetTools)
library(caret)

str(roi_data1)
input <- roi_data1[,2:8]
ideal <- class.ind(roi_data1$lc)
neurons <- 1
nnet_1 = nnet(input,ideal,size=neurons,softmax=TRUE,trace=FALSE)
plotnet(nnet_1)


## Prediction using neural network
ls8_pred_nnet1 <- predict(ls8_class, model=nnet_1, na.rm=T)
plot(ls8_pred_nnet1, col=colors1)

#test error due to 0 1 probability so add type=class
summary(roi_data1[,9])
nn1_prediction <- predict(nnet_1, roi_data1[2:8], type="class")
summary(nn1_prediction)
table(nn1_prediction,roi_data1[,9])

confusionMatrix(table(nn1_prediction,roi_data1[,9]))

#works for continuous regression only
#plot(roi_data1[,9], nn1_prediction, col='blue', pch=16, ylab = "predicted rating NN", xlab = "real rating")
#abline(0,1)

# Calculate Root Mean Square Error (RMSE)
#RMSE.NN = (sum((datatest$rating - predict_testNN)^2) / nrow(datatest)) ^ 0.5


#########################################################################################
fitControl <- trainControl( ## 10-fold CV
  method = "repeatedcv",
  number = 10,
  repeats = 10)

set.seed(825)


svm12 <- train(lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data=roi_data1, 
                 method = "rf", 
                 trControl = fitControl,
                 verbose = FALSE)
svm12

gbmFit1 <- train(lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data=roi_data1, 
                 method = "gbm", 
                 trControl = fitControl,
                 verbose = FALSE)
gbmFit1

gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                        n.trees = (1:30)*50, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

nrow(gbmGrid)

set.seed(825)
gbmFit2 <- train(lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data=roi_data1, 
                 method = "gbm", 
                 trControl = fitControl, 
                 verbose = FALSE, 
                 tuneGrid = gbmGrid)
gbmFit2


#########################################################################################
kfoldcv <- trainControl(method="cv", number=10)
performance_metric <- "Accuracy"

set.seed(123)
#.water <- train(lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data=roi_data1, method="", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))

#Linear Discriminant Analysis (LDA)
lda.water <- train(lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data=roi_data1, method="lda", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))

#Classification and Regression Trees (CART)
rpart.water <- train(lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data=roi_data1, method="rpart", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))

#Support Vector Machines (SVM)
svm.water <- train(lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data=roi_data1, method="svmRadial", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))

# Random Forest
rf.water <- train(lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data=roi_data1, method="rf", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))

# Gradient Bosting
gbm.water <- train(lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data=roi_data1, method="gbm", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))

# NN
nnet.water <- train(lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data=roi_data1, method="nnet", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))

#Accuracy
results.water <- resamples(list(lda=lda.water, rpart=rpart.water,  svm=svm.water, rf=rf.water, gbm=gbm.water, nnet=nnet.water))
summary(results.water)
dotplot(results.water)

# Predict
#ls8_pred_ <- predict(ls8_class, model=.water, na.rm=T)
ls8_pred_lda <- predict(ls8_class, model=lda.water, na.rm=T)
ls8_pred_rpart <- predict(ls8_class, model=rpart.water, na.rm=T)
ls8_pred_svmRadial <- predict(ls8_class, model=svmRadial.water, na.rm=T)
ls8_pred_rf <- predict(ls8_class, model=rf.water, na.rm=T)
ls8_pred_gbm <- predict(ls8_class, model=gbm.water, na.rm=T)
ls8_pred_nnet <- predict(ls8_class, model=nnet.water, na.rm=T)

#Plot
#plot(ls8_pred_, col=colors)
plot(ls8_pred_lda, col=colors)

plot(ls8_pred_nnet, col=colors)

#Confusion Matrix
sink("LS8_CM_lda.txt"); confusionMatrix(table(ls8_pred_lda,roi_data1[,9])); sink()
sink("LS8_CM_rpart.txt"); confusionMatrix(table(ls8_pred_rpart,roi_data1[,9])); sink()





# write to a new geotiff file (single)
if (require(rgdal)) {
  aa <- writeRaster(ls8_pred_rf1, filename="LS8_RandF_Class_1.tif", format="GTiff", overwrite=TRUE)
  bb <- writeRaster(ls8_pred_rp1, filename="LS8_Rpart_Class_1.tif", format="GTiff", overwrite=TRUE)
  cc <- writeRaster(ls8_pred_svm1, filename="LS8_SVM_Class_1.tif", format="GTiff", overwrite=TRUE)
  dd <- writeRaster(ls8_pred_svm_after_tune1, filename="LS8_SVM_Class_tune_1.tif", format="GTiff", overwrite=TRUE)
  ee <- writeRaster(ls8_pred_svm_after_tune2, filename="LS8_SVM_Class_tune_2.tif", format="GTiff", overwrite=TRUE)
  ff <- writeRaster(ls8_pred_nnet1, filename="LS8_nnet_Class_1.tif", format="GTiff", overwrite=TRUE)
}

#########################################################################################


#end cluster
stopCluster(cl)
