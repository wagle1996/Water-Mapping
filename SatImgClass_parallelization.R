# parallelizationDemo.R 
# A short code illustrating how parallelization can work in R for landcover mapping.
# Mostly borrowed (stolen) from the CRAN vignette and Ashotn Shortridge codes.
# written by TD Acharya, 8/June/2019


#http://ceholden.github.io/open-geo-tutorial/R/chapter_5_classification.html

setwd("D:/ECSA5/") #set directory
getwd()
list.files()

#######################################################################################
library(lattice, ggplot2)
library(dplyr)
library(sf)
library(caret)
library(rgdal)
library(raster)
library(reshape2)
#######################################################################################
# parallelizationDemo
library(doParallel) #Foreach Parallel Adaptor 
library(foreach)    #Provides foreach looping construct
detectCores()  # How many cores are out there?

#Define how many cores you want to use
cl <- makeCluster(detectCores()-2) # Never use all the cores!
#Register CoreCluster
registerDoParallel(cl)

#measure start time
start.time <- Sys.time()

## When you are done:
stopCluster(cl)

#measure end time
end.time <- Sys.time()

#measure difference in time
time.taken <- end.time - start.time
print(time.taken)   # compare with just %do% instead of %dopar%, or set cores to 1

#########################################################################################

#read tif file from landsat
ls8 <- brick('Full.tif')
names(ls8)
ls81<- stack('Full.tif')
names(ls81)

#rename Landsat bands
ls8_class <- ls8
names(ls8_class)
names(ls8_class) <- c('b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7')
names(ls8_class)

#plots indiviual layers
#plot(ls8) #orginal ls8 class names
#plot(ls8_class) # new ls8 calss names

#Plot RGB
#plotRGB(ls8_class, r=4, g=3, b=2, stretch="lin")
#plotRGB(ls8_class, r=5, g=3, b=2, stretch="lin")

#Save RGB


#########################################################################################
#https://www.neonscience.org/dc-shapefile-attributes-r

#read and prep shp file as training data
training1 <- read_sf(dsn='Random_pointsonly.shp', layer='Random_pointsonly')
training2 <- readOGR("Random_pointsonly.shp")

# ???
str(training2)

# view crs - note - this only works with the raster package loaded
crs(training2)

# view extent- note - this only works with the raster package loaded
extent(training2)

# view class
class(training2)
class(training2$class) # It is a factor variable - good!

# view features count
length(training2)
summary(training2$class)

# view metadata summary
training2
#########################################################################################
# Create color map

#Simple Colors
#ColorPalette <- c("blue","red")
#ColorPalette

#Mixed Colors
colors <- c(rgb(211, 211, 211, maxColorValue=255),  # Nonwater
            rgb(0, 0, 255, maxColorValue=255))      # Water

colors

colors1 <- c(rgb(0, 0, 255, maxColorValue=255),          # Nonwater
             rgb(211, 211, 211, maxColorValue=255))      # Water

#########################################################################################

#simply plot training data
#plot(training1)

#plot points only
#plot(training1, pch=16, col="green", add=TRUE)

#plot types of class
plotRGB(ls8_class, r=4, g=3, b=2, stretch="lin")
plot(training1, pch=16, col=colors, add=TRUE)

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
roi_data1$lc <- as.factor(training1$class)
#roi_data1$desc <- as.factor(training1$class)

#########################################################################################
#MUST SPLIT TRAIN and TEST later
#https://www.analyticsvidhya.com/blog/2017/09/creating-visualizing-neural-network-in-r/

#kfoldcv <- trainControl(method="repeatedcv", number=10)
kfoldcv <- trainControl(method="cv", number=10)
performance_metric <- "Accuracy"
#performance_metric <- "ROC" #Requires class probability

#########################################################################################
set.seed(333)

#.water <- train(lc ~ b1 + b2 + b3 + b4 + b5 + b6 + b7, data=roi_data1, method="", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))
#trellis.par.set(caretTheme())
#plot(.water)

names(roi_data1)    # Looks good! But let's get rid of the first field.
roi_data1 <- roi_data1[,2:ncol(roi_data1)]   # It's an ID field only, not useful.
names(roi_data1)

# Set up the prediction formula. It is just like an lm() formula.
# response variable is on the left of ~, predictor variables on the right.
excludeNames <- c('lc')
includeNames <- names(roi_data1)[!(names(roi_data1) %in% excludeNames)]
fnames <- paste(includeNames, collapse='+') 
frm <- as.formula(paste("lc ~ ", fnames))
frm    # Just a string with the formula!

my_mlms <- c('naive_bayes', 'rpart', 'svmRadial', 'rf', 'gbm', 'nnet')

print (my_mlms[1])
print (paste(my_mlms[1], "water", sep='_'))
mlm <- (my_mlms[1])
print(mlm)

#paste(mlm, "water", sep='_') <- train(frm, data=roi_data1, method= mlm, metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))
#trellis.par.set(caretTheme())
#plot(paste(mlm, "water", sep='_'))
#plot(paste(mlm, "water", sep='_'), metric = "Kappa")
#trellis.par.set(caretTheme())
#densityplot(paste(mlm, "water", sep='_'), pch = "|")

for (mlm in my_mlms){
  print(paste("The ML is", mlm))
  paste(mlm, "water", sep='_') <- train(frm, data=roi_data1, method=mlm, metric=performance_metric, trControl=kfoldcv, preProcess=c("center", "scale"))
  trellis.par.set(caretTheme())
  plot(paste(mlm, "water", sep='_'))
  plot(paste(mlm, "water", sep='_'), metric = "Kappa")
  trellis.par.set(caretTheme())
  densityplot(paste(mlm, "water", sep='_'), pch = "|")
}


rfFit <- train(frm, data = trainPts@data, method="rf", 
               metric=metric, tuneGrid=tunegrid, trControl=trctrl)

#Accuracy
results.water <- resamples(list(nb=nb.water, rpart=rpart.water,  svm=svmRadial.water, rf=rf.water, gbm=gbm.water, nnet=nnet.water))
summary(results.water)
sink("LS8_Summary.txt"); summary(results.water); sink()
#dotplot(results.water, metric = "Accuracy")
ggplot(results.water)
dotplot(results.water)
bwplot(results.water)
#xyplot(results.water) #works only with first two comparison
splom(results.water)

#Model Differences
difValues <- diff(results.water)
summary(difValues)
dotplot(difValues)
bwplot(difValues)

# Predict
#ls8_pred_ <- predict(ls8_class, model=.water, na.rm=T)
ls8_pred_nb <- predict(ls8_class, model=nb.water, na.rm=T)
ls8_pred_rpart <- predict(ls8_class, model=rpart.water, na.rm=T)
ls8_pred_svmRadial <- predict(ls8_class, model=svmRadial.water, na.rm=T)
ls8_pred_rf <- predict(ls8_class, model=rf.water, na.rm=T)
ls8_pred_gbm <- predict(ls8_class, model=gbm.water, na.rm=T)
ls8_pred_nnet <- predict(ls8_class, model=nnet.water, na.rm=T)

#Plot
#plot(ls8_pred_, col=colors)
#plot(ls8_pred_nb, col=colors)
#plot(ls8_pred_rpart, col=colors)
#plot(ls8_pred_svmRadial, col=colors)
#plot(ls8_pred_rf, col=colors)
#plot(ls8_pred_gbm, col=colors)
#plot(ls8_pred_nnet, col=colors)

#Confusion Matrix
#sink("LS8_CM_.txt"); confusionMatrix(table(predict(.water, roi_data1[2:8]),roi_data1[,9])); sink()
sink("LS8_CM_nb.txt"); confusionMatrix(table(predict(nb.water, roi_data1[2:8]),roi_data1[,9])); sink()
sink("LS8_CM_rpart.txt"); confusionMatrix(table(predict(rpart.water, roi_data1[2:8]),roi_data1[,9])); sink()
sink("LS8_CM_svmRadial.txt"); confusionMatrix(table(predict(svmRadial.water, roi_data1[2:8]),roi_data1[,9])); sink()
sink("LS8_CM_rf.txt"); confusionMatrix(table(predict(rf.water, roi_data1[2:8]),roi_data1[,9])); sink()
sink("LS8_CM_gbm.txt"); confusionMatrix(table(predict(gbm.water, roi_data1[2:8]),roi_data1[,9])); sink()
sink("LS8_CM_nnet.txt"); confusionMatrix(table(predict(nnet.water, roi_data1[2:8]),roi_data1[,9])); sink()

# write to a new geotiff file (single)
if (require(rgdal)) {
  writeRaster(ls8_pred_nb, filename="LS8_pred_nb.tif", format="GTiff", overwrite=TRUE)
  writeRaster(ls8_pred_rpart, filename="ls8_pred_rpart.tif", format="GTiff", overwrite=TRUE)
  writeRaster(ls8_pred_svmRadial, filename="LS8_svmRadial.tif", format="GTiff", overwrite=TRUE)
  writeRaster(ls8_pred_rf, filename="LS8_rf.tif", format="GTiff", overwrite=TRUE)
  writeRaster(ls8_pred_gbm, filename="LS8_gbm.tif", format="GTiff", overwrite=TRUE)
  writeRaster(ls8_pred_nnet, filename="LS8_nnet.tif", format="GTiff", overwrite=TRUE)
}