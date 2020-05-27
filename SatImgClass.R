#http://ceholden.github.io/open-geo-tutorial/R/chapter_5_classification.html


setwd("D:/ECSA5/") #set directory
getwd()
list.files()

#######################################################################################

#library(parallel)
#detectCores()
#numCores <- detectCores()
#numCores
#cl <- makeCluster(numCores)

#clusterEvalQ(cl, {
#  library(randomForest)
#  library(raster)
#  library(rgdal)
#  library(sf)
#})


#install.packages("neuralnet")
#install.packages("caret", repos="http://cran.rstudio.com/", dependencies=TRUE)
#install.packages("Biobase", repos="http://cran.rstudio.com/", dependencies=TRUE)
#library(car)


#if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")
#BiocManager::install("Biobase", version = "3.8")


library(raster)
library(rgdal)
library(caret)
library(lattice, ggplot2)
library(neuralnet)
library(ggplot2)
library(nnet)
library(dplyr)
library(reshape2)
library("e1071")

#######################################################################################################

ls8 <- brick('Full.tif')
names(ls8)

#rename Landsat bands
ls8_class <- ls8
names(ls8_class)
names(ls8_class) <- c('b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7')
names(ls8_class)
plot(ls8_class)
plotRGB(ls8_class, r=4, g=3, b=2, stretch="lin")
plotRGB(ls8_class, r=5, g=3, b=2, stretch="lin")

#not useful
#library(rgdal)
#training0 <- readOGR(dsn='Random_pointsonly.shp', layer='Random_pointsonly')
#plot(training0)

#read and prep training data
library(sf)
training1 <- read_sf(dsn='Random_pointsonly.shp', layer='Random_pointsonly')
plot(training1)

roi_data1 <- extract(ls8_class, training1, df=TRUE)
roi_data1copy <- extract(ls8_class, training1, df=TRUE)

#Remember that this dataset may have clouds or cloud shadows. Let's mask them out:
#roi_data[which(roi_data$Band.8 > 1)] <- NA

#We'll also need to attach the labels to this DataFrame:
roi_data1$lc <- as.factor(training1$ID)
roi_data1$desc <- as.factor(training1$class)


#MUST SPLIT TRAIN and TEST later
#https://www.analyticsvidhya.com/blog/2017/09/creating-visualizing-neural-network-in-r/


#################################################################################################

# Set seed value for RBG for reproducibility
set.seed(1234567890)
colnames(roi_data1)

# Shorten column names
#colnames(roi_data1) <- c('ID', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'lc', 'desc')
#colnames(roi_data1)

# Create color map
colors <- c(rgb(211, 211, 211, maxColorValue=255),  # Urban
            rgb(0, 0, 255, maxColorValue=255))      # Water

# Create color map
colors1 <- c(rgb(0, 0, 255, maxColorValue=255),          # Urban
             rgb(211, 211, 211, maxColorValue=255))      # Water



###############################################################################################
#Unsupervised not working
#kMeansResult <- kmeans(ls8[], centers=6)
#result <- raster(ls8[[1]])
#result <- setValues(result, kMeansResult$cluster)
#plot(result)

###############################################################################################

#RANDOM FOREST
library(randomForest)
rf1 <- randomForest(lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data=roi_data1, importance=TRUE)
summary(rf1)

# Predict!
ls8_pred_rf1 <- predict(ls8_class, model=rf1, na.rm=T)
plot(ls8_pred_rf1, col=colors)

#test
summary(roi_data1[,9])
rf1_prediction <- predict(rf1, roi_data1[2:8])
summary(rf1_prediction)
table(rf1_prediction,roi_data1[,9])

confusionMatrix(table(rf1_prediction,roi_data1[,9]))

#stats
print(rf1)
varImpPlot(rf1)

############################################################################################

#SVM
library("e1071")
svm1 <- svm(lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data=roi_data1)
summary(svm1)

#prediction
ls8_pred_svm1 <- predict(ls8_class, model=svm1, na.rm=T)
plot(ls8_pred_svm1, col=colors)

#test
summary(roi_data1[,9])
svm1_prediction <- predict(svm1, roi_data1[2:8])
summary(svm1_prediction)
table(svm1_prediction,roi_data1[,9])

confusionMatrix(table(svm1_prediction,roi_data1[,9]))


#SVM_tuning
svm_tune1 <- tune(svm, lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data=roi_data1, kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))
summary(svm_tune1)
print(svm_tune1)

#prediction
svm_model_after_tune1 <- svm(lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data=roi_data1, kernel="radial", cost=1, gamma=0.5)
ls8_pred_svm_after_tune1 <- predict(ls8_class, model=svm_model_after_tune1, na.rm=T)
plot(ls8_pred_svm_after_tune1, col=colors)

#test
summary(roi_data1[,9])
svm_tune1_prediction <- predict(svm_model_after_tune1, roi_data1[2:8])
summary(svm_tune1_prediction)
table(svm_tune1_prediction,roi_data1[,9])

confusionMatrix(table(svm_tune1_prediction,roi_data1[,9]))

#SVM_tuning
svm_tune2 <- tune.svm(lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data=roi_data1, gamma = 10^(-5:-1), cost = 10^(-3:1))
summary(svm_tune2)
print(svm_tune2)

#prediction
svm_model_after_tune2 <- svm(lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data=roi_data1, kernel="radial", cost=1, gamma=0.01)
ls8_pred_svm_after_tune2 <- predict(ls8_class, model=svm_model_after_tune2, na.rm=T)
plot(ls8_pred_svm_after_tune2, col=colors)

#test
summary(roi_data1[,9])
svm_tune2_prediction <- predict(svm_model_after_tune2, roi_data1[2:8])
summary(svm_tune2_prediction)
table(svm_tune2_prediction,roi_data1[,9])

confusionMatrix(table(svm_tune2_prediction,roi_data1[,9]))

#stats


#####################################################################################################
#Recursive Partitioning and Regression Trees
library(rpart)
rp1 <- rpart(lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data=roi_data1)
summary(rp1)

# Predict!
ls8_pred_rp1 <- predict(ls8_class, model=rp1, na.rm=T)
plot(ls8_pred_rp1, col=colors1)

#test error due to 0 1 probability so add type=class
summary(roi_data1[,9])
rp1_prediction <- predict(rp1, roi_data1[2:8], type="class")
summary(rp1_prediction)
table(rp1_prediction,roi_data1[,9])

confusionMatrix(table(rp1_prediction,roi_data1[,9]))

#tree
print(rp1)
plot(rp1)
text(rp1)

#Stats
printcp(rp1)
plotcp(rp1)

par(mfrow=c(1,2)) 
rsq.rpart(rp1) # cross-validation results 

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


#############################################################################################################


# write to a new geotiff file (single)
if (require(rgdal)) {
  aa <- writeRaster(ls8_pred_rf1, filename="LS8_RandF_Class_1.tif", format="GTiff", overwrite=TRUE)
  bb <- writeRaster(ls8_pred_rp1, filename="LS8_Rpart_Class_1.tif", format="GTiff", overwrite=TRUE)
  cc <- writeRaster(ls8_pred_svm1, filename="LS8_SVM_Class_1.tif", format="GTiff", overwrite=TRUE)
  dd <- writeRaster(ls8_pred_svm_after_tune1, filename="LS8_SVM_Class_tune_1.tif", format="GTiff", overwrite=TRUE)
  ee <- writeRaster(ls8_pred_svm_after_tune2, filename="LS8_SVM_Class_tune_2.tif", format="GTiff", overwrite=TRUE)
  ff <- writeRaster(ls8_pred_nnet1, filename="LS8_nnet_Class_1.tif", format="GTiff", overwrite=TRUE)
}


# write to a new geotiff file (multi)
#if (require(rgdal)) {
#  aaa <- writeRaster(ls8_pred_rp1, filename="multi.tif", options="INTERLEAVE=BAND", overwrite=TRUE)
#}



#https://www.r-bloggers.com/fitting-a-neural-network-in-r-neuralnet-package/

## Scale data for neural network
#max = apply(roi_data1 , 2 , max)
#min = apply(roi_data1, 2 , min)
#scaled = as.data.frame(scale(roi_data1, center = min, scale = max - min))

#m <- model.matrix(~ lc + b1 + b2 + b3 + b4 + b5 + b6 + b7, data = roi_data1)
#head(m)
#r <- neuralnet(lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data=roi_data1, hidden=10, threshold=0.01)

#NN = neuralnet(lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, roi_data1, hidden = 3 , linear.output = T )

#f <- as.formula("lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7")
#nn <- neuralnet(f, data=roi_data1, hidden=c(5,3), linear.output=T)

#n_net_ <- neuralnet(lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data=roi_data1, hidden = c(16, 12), act.fct = "tanh", linear.output = FALSE)
#plot(n_net)


#library(caret)
#my.grid <- expand.grid(.decay = c(0.5, 0.1), .size = c(5, 6, 7))
#prestige.fit <- train(lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data = roi_data1, method = "nnet", maxit = 1000, tuneGrid = my.grid, trace = F, linout = 1)  
