#RDataMining-book.pdf
#
setwd("/Users/dbiswas/Documents/MS/ds_study/ms/R")

#1.3.1 The Iris Dataset
str(iris)

data("bodyfat", package = "TH.data")
str(bodyfat)

#CH2 - Data Import and Export

#2.1 Save and Load R Data
a <- 1:10
b <- letters[1:5]
save(a, b, file="data/mydatafile.Rdata")
rm(a, b)
load("data/mydatafile.Rdata")
print(a)

#Save as RDS file
a <- 1:10
saveRDS(a, file="data/mydatafile2.rds")
a2 <- readRDS("./data/mydatafile2.rds")
print(a2)

#R also provides function save.image() to save everything in current workspace into a single
#le, which is very convenient to save your current work and resume it later, if the data loaded into
#R are not very big.

#2.4.1 Read from Databases
# library(RODBC)
# connection <- odbcConnect(dsn="servername",uid="userid",pwd="******")
# query <- "SELECT * FROM lib.table WHERE ..."
# # or read query from file
#    # query <- readChar("data/myQuery.sql", nchars=99999)
#    myData <- sqlQuery(connection, query, errors=TRUE)
#  odbcClose(connection)

#There are also sqlSave() and sqlUpdate() for writing or updating a table in an ODBC database.
#While package RODBC can read and write EXCEL les on Windows, but it does not work
#directly on Mac OS X, because an ODBC driver for EXCEL is not provided by default on Mac.

#2.4.2 Output to and Input from EXCEL Files
#  library(RODBC)
#  filename <- "data/myExcelFile.xls"
#  xlsFile <- odbcConnectExcel(filename, readOnly = FALSE)
#  sqlSave(xlsFile, a, rownames = FALSE)
#  b <- sqlFetch(xlsFile, "sheetname")
#  odbcClose(xlsFile)

#2.5 Read and Write EXCEL les with package xlsx

library(xlsx)
table(iris$Species)
 setosa <- subset(iris, Species == "setosa")
 
#row,names=T will add the row number as the first column
 write.xlsx(setosa, file="./data/iris.xlsx", sheetName="setosa", row.names=F)
 versicolor <- subset(iris, Species == "versicolor")
#append=T adds another sheet in the same xls. If not set, the xls itself will be overwritten
 write.xlsx(versicolor, file="./data/iris.xlsx", sheetName="versicolor", row.names=F, append=T)
 virginica <- subset(iris, Species == "virginica")
 write.xlsx(virginica, file="./data/iris.xlsx", sheetName="virginica", row.names=F, append=T)
 a <- read.xlsx("./data/iris.xlsx", sheetName="setosa")
 head(a)


####CH3 - Data Exploration and Visualization ###

#3.1 Have a Look at Data
dim(iris)
str(iris)
names(iris)
attributes(iris)


## draw a sample of 5 rows
idx <- sample(1:nrow(iris), 5)
idx <- sample(nrow(iris), 5)

#Note : For sample the default for size is the number of items inferred from the first argument, 
#so that sample(x) generates a random permutation of the elements of x (or 1:x).


#3.2 Explore Individual Variables
summary(iris)
quantile(iris$Sepal.Length)

# 0% 25% 50% 75% 100%
# 4.3 5.1 5.8 6.4 7.9

quantile(iris$Sepal.Length, c(0.1, 0.3, 0.65))
# 10% 30% 65%
# 4.80 5.27 6.20

var(iris$Sepal.Length)
hist(iris$Sepal.Length)
plot(density(iris$Sepal.Length))

#[2016-02-20]
#3.3 Explore Multiple Variables
#After checking the distributions of individual variables, we then investigate the relationships between
#two variables. Below we calculate covariance and correlation between variables with cov()
#and cor().
cov(iris$Sepal.Length, iris$Petal.Length)
cor(iris$Sepal.Length, iris$Petal.Length)

cov(iris[,1:4])
cor(iris[,1:4])

#Correlation shows whether and how strongly a pair of variables are related to each other. It ranges
#from -1 to +1. The closer the correlation is to +1 (or -1), the more strongly the two variables are
#positively (or negatively) related. When it is close to zero, it means that there is no relationship
#between them.

#Next, we compute the stats of Sepal.Length of every Species with aggregate().
aggregate(Sepal.Length ~ Species, summary, data=iris)

#boxplot() show the median, rst and third quartile of a distribution (i.e., the 50%, 25% and 75% points 
#in cumulative distribution), and outliers.
boxplot(Sepal.Length ~ Species, data=iris, xlab="Species", ylab="Sepal.Length")

#A scatter plot can be drawn for two numeric variables with plot() as below. Using function
#with(), we don't need to add \iris$" before variable names.
#pch : symbol
with(iris, plot(Sepal.Length, Sepal.Width, col=Species, pch=as.numeric(Species)))
plot(iris$Sepal.Length, iris$Sepal.Width, col=iris$Species, pch=as.numeric(iris$Species))

#We can use jitter() to add a small amount of noise to the data before plotting.
plot(jitter(iris$Sepal.Length), jitter(iris$Sepal.Width))

#A smooth scatter plot can be plotted with function smoothScatter(), which a smoothed color
#density representation of the scatterplot, obtained through a kernel density estimate.
 smoothScatter(iris$Sepal.Length, iris$Sepal.Width)

#A matrix of scatter plots can be produced with function pairs()
pairs(iris)

#A 3D scatter plot can be produced with package scatterplot3d
library(scatterplot3d)
scatterplot3d(iris$Petal.Width, iris$Sepal.Length, iris$Sepal.Width)


#Package rgl supports interactive 3D scatter plot with plot3d().
#install.packages("rgl")
#library(rgl)
#plot3d(iris$Petal.Width, iris$Sepal.Length, iris$Sepal.Width)


#A heat map presents a 2D display of a data matrix, which can be generated with heatmap()
#in R. With the code below, we calculate the similarity between different 
#flowers in the iris data with dist() and then plot it with a heat map.

#Species column removed by iris[,1:4])
distMatrix <- as.matrix(dist(iris[,1:4]))
heatmap(distMatrix)

#A level plot can be produced with function levelplot() in package lattice. 
#Function grey.colors() creates a vector of gamma-corrected gray colors. 
#rainbow(), which creates a vector of contiguous colors.
library(lattice)
levelplot(Petal.Width~Sepal.Length*Sepal.Width, iris, cuts=9,col.regions=grey.colors(10)[10:1])

#Contour plots can be plotted with contour() and filled.contour() in package graphics, and
#with contourplot() in package lattice.
filled.contour(volcano, color=terrain.colors, asp=1, plot.axes=contour(volcano, add=T))

#Another way to illustrate a numeric matrix is a 3D surface plot shown as below, which is 
#generated with function persp().
persp(volcano, theta=25, phi=30, expand=0.5, col="lightblue")

#Parallel coordinates provide nice visualization of multiple dimensional data. A parallel 
#coordinates plot can be produced with parcoord() in package MASS, and with parallelplot() 
#in package lattice
library(MASS)
parcoord(iris[1:4], col=iris$Species)

library(lattice)
parallelplot(~iris[1:4] | Species, data=iris)


#Package ggplot2 [Wickham, 2009] supports complex graphics, which are very useful for 
#exploring data.

library(ggplot2)
qplot(Sepal.Length, Sepal.Width, data=iris, facets=Species ~.)

#3.5 Save Charts into Files
#If there are many graphs produced in data exploration, a good practice is to save them into les.
#R provides a variety of functions for that purpose. Below are examples of saving charts into PDF
#and PS les respectively with functions pdf() and postscript(). Picture les of BMP, JPEG,
#PNG and TIFF formats can be generated respectively with bmp(), jpeg(), png() and tiff().
#Note that the les (or graphics devices) need be closed with graphics.off() or dev.off() after
#plotting.
 # save as a PDF file
   pdf("myPlot.pdf")
 x <- 1:50
 plot(x, log(x))
 graphics.off()
 #
   # Save as a postscript file
   postscript("myPlot2.ps")
 x <- -20:20
 plot(x, x^2)
 graphics.off()






#CH-4 : Decision Trees and Random Forest
########################################

# packages : party, rpart and randomForest

##party example##

#Note:
#sample(2, nrow(iris)
#It will create a vector of 150 elements i.e. nrow(iris) and each element will have value either 1 or 2.
#The ratio of 1 and 2 wil be 70:30.
#str(ind)
#int [1:150] 2 1 1 1 2 2 1 1 1 1 ...
#
#If replace=FALSE, then the interpretation of the parameter changes. First parameter indicates
#total number of observations and 2nd parameter indicates how many observations to take as a sample

ind <- sample(2, nrow(iris), replace=TRUE, prob=c(0.7, 0.3))

#Vector from previous step is used to filter out iris rows into training and test dataset.
trainData <- iris[ind==1,]
testData <- iris[ind==2,]
library(party)
myFormula <- Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width
iris_ctree <- ctree(myFormula, data=trainData)
# check the prediction
table(predict(iris_ctree), trainData$Species)
print(iris_ctree)
plot(iris_ctree)
plot(iris_ctree, type="simple")

#they are shown as \y" in leaf nodes. For example, node 2 is labeled with \n=40, y=(1, 0, 0)", 
#which means that it contains 40 training instances and all of them belong to the rest class \setosa".

# predict on test data
testPred <- predict(iris_ctree, newdata = testData)
table(testPred, testData$Species)


#4.2 Decision Trees with Package rpart

data("bodyfat", package = "TH.data")
dim(bodyfat)

attributes(bodyfat)
#$names
#$row.names

set.seed(1234)
ind <- sample(2, nrow(bodyfat), replace=TRUE, prob=c(0.7, 0.3))
bodyfat.train <- bodyfat[ind==1,]
bodyfat.test <- bodyfat[ind==2,]
# train a decision tree
library(rpart)
myFormula <- DEXfat ~ age + waistcirc + hipcirc + elbowbreadth + kneebreadth
#minsplit : the minimum number of observations that must exist in a node in order for a split to be attempted.
bodyfat_rpart <- rpart(myFormula, data = bodyfat.train, control = rpart.control(minsplit = 10))
attributes(bodyfat_rpart)

#$names
#$xlevels
#$class

#cptable : complexity parameter
#It serves as a penalty term to control tree size and is always monotonic with the number of splits (nsplit). 
#The smaller the value of CP, the more complex will be the tree (the greater the number of splits).
#Other parameters of cptable
#rel error : For a regression tree, the relative error (rel error) is the average deviance of the current 
#tree divided by the average deviance of the null tree.
#xerror : The cross-validation error (xerror) is based on a 10-fold cross-validation and is again measured 
#relative to the deviance of the null model. As expected the cross-validation error is greater than the relative error. 
#Using the same data to both fit and test a model results in over-optimistic fit diagnostics.

##Something to note for further reference
#http://www.unc.edu/courses/2010spring/ecol/562/001/docs/lectures/lecture22.htm
#
#While relative error is guaranteed to decrease as the tree gets more complex, this will not normally be the case for 
#cross-validation error. Because the cross-validation error is still decreasing in the output shown above, the default 
#tree size is probably too small. We need to refit the model and force rpart to carry out additional splits. This can be 
#accomplished with the control argument of rpart. I use the rpart.control function to specify a value for cp= that is 
#smaller than the default value of 0.01. Because the cross-validation error results are random, I use the set.seed 
#function first to set the seed for the random number stream so that the results obtained are reproducible.
#set.seed(20)
#parrot_tree2 <- rpart(Parrot ~ CoralTotal + as.factor(Month) + as.factor(Station) + as.factor(Method), data = Bahama, control=rpart.control(cp=.001))

print(bodyfat_rpart$cptable)

#Columns you get when rpart model is printed as below
#n= 56
#node), split, n, deviance, yval
#* denotes terminal node
print(bodyfat_rpart)

plot(bodyfat_rpart)
#Not sure what use.n=T does?
text(bodyfat_rpart, use.n=T)

#Then we select the tree with the minimum prediction error
#In our case, it generated 8 trees. We are selecting the tree with least xerror
opt <- which.min(bodyfat_rpart$cptable[,"xerror"])
cp <- bodyfat_rpart$cptable[opt, "CP"]
cp <- bodyfat_rpart$cptable[8, "CP"]
bodyfat_prune <- prune(bodyfat_rpart, cp = cp)
print(bodyfat_prune)

plot(bodyfat_prune)
text(bodyfat_prune, use.n=T)


#After that, the selected tree is used to make prediction and the predicted values are compared
#with actual labels. In the code below, function abline() draws a diagonal line. The predictions
#of a good model are expected to be equal or very close to their actual values, that is, most points
#should be on or close to the diagonal line.

DEXfat_pred <- predict(bodyfat_prune, newdata=bodyfat.test)
xlim <- range(bodyfat$DEXfat)
#Plotting predicted vs actual value. e.g. For an observation, if predicted value is
#11, and actual value is also 11, then the point will fall on the diagonal line. If 
#there is any difference, it will be evident from this plot.
plot(DEXfat_pred ~ DEXfat, data=bodyfat.test, xlab="Observed", ylab="Predicted", ylim=xlim, xlim=xlim)

#y=a + bx
#with a=0, b=1 the line passes thru origin as y becomes x(y=x)
abline(a=0, b=1)



#4.3 Random Forest#

# k time repeat the following procedure
# - draw bootstrap sample from dataset
# - train decision tree
# Until tree is of maximum size
# Choose next leaf node
# select m attributes at random
# 
# - Measure out-of-bag error
# -When u drwa bootstrap, u might get duplicate. when duplicate, the othe one may have left out
# - evaluate the samples that were not selected in bootstrap
# - provides
# - measure of strength (inverse error rate)
# - correlation between trees(which increases the forest error rate)
# - variable importance
# 
# 
# *** make prediction by majority vote among the k trees

# Variable Importance
# 
# Gini Co-efficeient
# - measures inequality
# 
#
# Random Forest on Big Data
# - Easy to parrelelize (trees are build independently)
# - Handles 'small n big p' problems naturally
# 
# 
# Summary : Decision Trees and Forests
# 
# *** Represenation
# - Decision Trees 
# - Sets of decision trees with majority vote
# 
# *** Evaluation
# - Accuracy 
# - Random Forests : out-of-bag error (generated on ur training set)
# 
# *** Optimization
# - Information Gain or Gini Index
# 
# Adv.
# categorical attributes
# many attributes
# diff variations of them
# simple to interpret the result of
# simple to implement
# 
# 
# Pretty fantastic general purpose


#There are two limitations with function randomForest()
#First, it cannot handle data with missing values, and users have to impute data before 
#feeding them into the function.
#Second, there is a limit of 32 to the maximum number of levels of each categorical attribute.
#Attributes with more than 32 levels have to be transformed first before using randomForest().

#An alternative is cforest() from package party. categorical variables with more levels will take more
#cpu and memory

#The iris data is first split into two subsets: training (70%) and test (30%).
ind <- sample(2, nrow(iris), replace=TRUE, prob=c(0.7, 0.3))
trainData <- iris[ind==1,]
testData <- iris[ind==2,]

#"Species ~ .", which means to predict Species with all other variables in the data.

library(randomForest)
rf <- randomForest(Species ~ ., data=trainData, ntree=100, proximity=TRUE)
table(predict(rf), trainData$Species)
print(rf)

#Few importnat parameters when model is printed
#No. of variables tried at each split :
#OOB estimate of error rate : 
#Confusion matrix : 

attributes(rf)

#After that, we plot the error rates with various number of trees.
plot(rf)

#The importance of variables can be obtained with functions importance() and varImpPlot().
importance(rf)
varImpPlot(rf)

#Finally, the built random forest is tested on test data, and the result is checked with functions
#table() and margin(). The margin of a data point is as the proportion of votes for the correct
#class minus maximum proportion of votes for other classes. Generally speaking, positive margin means 
#correct classification.
irisPred <- predict(rf, newdata=testData)
table(irisPred, testData$Species)

plot(margin(rf, testData$Species))

#[2016-03-03]

#CH - 5 : Regression
####################

#rep(x, ...)
#rep.int(x, times)
year <- rep(2008:2010, each=4)
quarter <- rep(1:4, 3)
cpi <- c(162.2, 164.6, 166.5, 166.0,
         + 166.2, 167.0, 168.6, 169.5,
         + 171.0, 172.1, 173.3, 174.0)
plot(cpi, xaxt="n", ylab="CPI", xlab="")
#draw x-axis
#In the code below, an x-axis is added manually with function axis(), where las=3 makes text vertical.
axis(1, labels=paste(year,quarter,sep="Q"), at=1:12, las=3)
cor(year,cpi)
cor(quarter,cpi)

#Then a linear regression model is built with function lm() on the above data, using year and quarter as predictors and CPI as response.
fit <- lm(cpi ~ year + quarter)
attributes(fit)
# differences between observed values and fitted values
residuals(fit)
summary(fit)
plot(fit)
#predict
data2011 <- data.frame(year=2011, quarter=1:4)
cpi2011 <- predict(fit, newdata=data2011)
#Making a style vector. 1 will repeat 12 times, 2 will repeat 4 times
style <- c(rep(1,12), rep(2,4))
plot(c(cpi, cpi2011), xaxt="n", ylab="CPI", xlab="", pch=style, col=style)
axis(1, at=1:16, las=3,
       + labels=c(paste(year,quarter,sep="Q"), "2011Q1", "2011Q2", "2011Q3", "2011Q4"))


#5.2 Logistic Regression
#logit(y) = c0 + c1x1 + c2x2 +    + ckxk;
#where x1; x2;    ; xk are predictors, y is a response to predict, and logit(y) = ln( y
#                                                                                        1􀀀y ). The above
#equation can also be written as
#y =
#  1
#1 + e􀀀(c0+c1x1+c2x2++ckxk) :
#Logistic regression can be built with function glm() by setting family to binomial(link="logit").

#5.3 Generalized Linear Regression

data("bodyfat", package="TH.data")
myFormula <- DEXfat ~ age + waistcirc + hipcirc + elbowbreadth + kneebreadth
bodyfat.glm <- glm(myFormula, family = gaussian("log"), data = bodyfat)
#Can alternativelt try with family=poisson
#bodyfat.glm <- glm(myFormula, family = poisson, data = bodyfat)
summary(bodyfat.glm)
pred <- predict(bodyfat.glm, type="response")
plot(bodyfat$DEXfat, pred, xlab="Observed Values", ylab="Predicted Values")
abline(a=0, b=1)

#In the above code, if family=gaussian("identity") is used, the built model would be similar
#to linear regression. One can also make it a logistic regression by setting family to binomial("
#logit").

#Q. What is difference between Linear Regression and Generelized Linear Regression?
#While linear regression is to nd the line that comes closest to data, non-linear regression is to
#t a curve through data. Function nls() provides nonlinear regression.


#CH 6 - Clustering
################

#6.1 The k-Means Clustering
#library(cluster)
iris2 <- iris
iris2$Species <- NULL
(kmeans.result <- kmeans(iris2, 3))
table(iris$Species, kmeans.result$cluster)

plot(iris2[c("Sepal.Length", "Sepal.Width")], col = kmeans.result$cluster)
# plot cluster centers
points(kmeans.result$centers[,c("Sepal.Length", "Sepal.Width")], col = 1:3,
+ pch = 8, cex=2)


#6.2. THE K-MEDOIDS CLUSTERING

#As an enhanced version of pam(), function pamk() in package fpc [Hennig, 2015]
#does not require a user to choose k

library(fpc)
pamk.result <- pamk(iris2)
# number of clusters
pamk.result$nc
# check clustering against actual species
table(pamk.result$pamobject$clustering, iris$Species)
layout(matrix(c(1,2),1,2)) # 2 graphs per page
plot(pamk.result$pamobject)
layout(matrix(1)) # change back to one graph per page



#6.3 Hierarchical Clustering
#We first draw a sample of 40 records from the iris data, so that the clustering plot will not be
#over crowded.
idx <- sample(1:dim(iris)[1], 40)
irisSample <- iris[idx,]
irisSample$Species <- NULL
hc <- hclust(dist(irisSample), method="ave")
plot(hc, hang = -1, labels=iris$Species[idx])
# cut tree into 3 clusters
rect.hclust(hc, k=3)
groups <- cutree(hc, k=3)


#6.4 Density-based Clustering

#hclust(d, method = "complete", members = NULL)
#d  
#a dissimilarity structure as produced by dist.

#method	
#the agglomeration method to be used. This should be (an unambiguous abbreviation of) one of "ward.D", "ward.D2", "single", "complete", "average" (= UPGMA), "mcquitty" (= WPGMA), "median" (= WPGMC) or "centroid" (= UPGMC).

idx <- sample(1:dim(iris)[1], 40)
irisSample <- iris[idx,]
irisSample$Species <- NULL
hc <- hclust(dist(irisSample), method="ave")
#Easy way to put label
plot(hc, hang = -1, labels=iris$Species[idx])
# cut tree into 3 clusters
rect.hclust(hc, k=3)
groups <- cutree(hc, k=3)


#6.4 Density-based Clustering
#Leaving it for now



#CH - 7 Outlier Detection
##########################
#attach() : Attach Set of R Objects to Search Path

#7.1 Univariate Outlier Detection
set.seed(3147)
x <- rnorm(100)
summary(x)
# outliers
boxplot.stats(x)$out
boxplot(x)

y <- rnorm(100)
df <- data.frame(x, y)
#confusing step; it's just removing the variable declaration from working area
rm(x, y)
head(df)
#Make it searchble
attach(df)

# find the index of outliers from x
(a <- which(x %in% boxplot.stats(x)$out))
# find the index of outliers from y
(b <- which(y %in% boxplot.stats(y)$out))
detach(df)
#outliers in both x and y
(outlier.list1 <- intersect(a,b))
plot(df)
points(df[outlier.list1,], col="red", pch="+", cex=2.5)

#Similarly, we can also take outliers as those data which are outliers in either x or y.
# outliers in either x or y
(outlier.list2 <- union(a,b))
plot(df)
points(df[outlier.list2,], col="blue", pch="x", cex=2)

#7.2 Outlier Detection with LOF

#LOF (Local Outlier Factor).
#LOF (Local Outlier Factor) is an algorithm for identifying density-based local outliers.
#With LOF, the local density of a point is compared with that of its neighbors.
#If the former is significantly lower than the latter (with an LOF value greater than one), the point 
#is in a sparser region than its neighbors, which suggests it be an outlier. A shortcoming of LOF
#is that it works on numeric data only.

#Rest - Leaving it for now

#7.3 Outlier Detection by Clustering
#Another way to detect outliers is clustering. By grouping data into clusters, those data not
#assigned to any clusters are taken as outliers.
#We can also detect outliers with the k-means algorithm. With k-means, the data are partitioned
#into k groups by assigning them to the closest cluster centers. After that, we can calculate the
#distance (or dissimilarity) between each object and its cluster center, and pick those with largest
#distances as outliers.
# remove species from the data to cluster
iris2 <- iris[,1:4]
kmeans.result <- kmeans(iris2, centers=3)
# cluster centers
kmeans.result$centers
# cluster IDs
kmeans.result$cluster
## calculate distances between objects and cluster centers

#Associating each record to one of the centers
centers <- kmeans.result$centers[kmeans.result$cluster, ]
#Find distance of each point from center
#rowSums : Form Row and Column Sums and Means
distances <- sqrt(rowSums((iris2 - centers)^2))
# pick top 5 largest distances
outliers <- order(distances, decreasing=T)[1:5]
# who are outliers
print(outliers)
print(iris2[outliers,])
# plot clusters
#col = color
plot(iris2[,c("Sepal.Length", "Sepal.Width")], pch="o", col=kmeans.result$cluster, cex=0.3)
# plot cluster centers
points(kmeans.result$centers[,c("Sepal.Length", "Sepal.Width")], col=1:3,pch=8, cex=1.5)
# plot outliers
points(iris2[outliers, c("Sepal.Length", "Sepal.Width")], pch="+", col=4, cex=1.5)


#7.4 Outlier Detection from Time Series
#the time series data are first decomposed with robust regression using function stl() and then outliers
#are identified.
> # use robust fitting
f <- stl(AirPassengers, "periodic", robust=TRUE)
(outliers <- which(f$weights<1e-8))

# set layout
op <- par(mar=c(0, 4, 0, 3), oma=c(5, 0, 4, 0), mfcol=c(4, 1))
plot(f, set.pars=NULL)
sts <- f$time.series
# plot outliers
points(time(sts)[outliers], 0.8*sts[,"remainder"][outliers], pch="x", col="red")
par(op) # reset layout


#CH 8 - Time Series Analysis and Mining
#autoregressive integrated moving average (ARIMA)
#Dynamic Time Warping (DTW)
#DWT (Discrete Wavelet Transform)

#8.1 Time Series Data in R
#A frequency of 7 indicates that a time series is composed of weekly data, and 12 and 4 are used respectively for
#monthly and quarterly series.

> a <- ts(1:30, frequency=12, start=c(2011,3))
> print(a)
> str(a)
> attributes(a)

#8.2 Time Series Decomposition

# decompose time series
apts <- ts(AirPassengers, frequency=12)
f <- decompose(apts)
#apts <- ts(AirPassengers, frequency=12)
f <- stats::decompose(apts)
> # seasonal figures
f$figure
plot(f$figure, type="b", xaxt="n", xlab="")
# get names of 12 months in English words
monthNames <- months(ISOdate(2011,1:12,1))
# label x-axis with month names
# las is set to 2 for vertical label orientation
axis(1, at=1:12, labels=monthNames, las=2)
plot(f)

#8.3 Time Series Forecasting
fit <- arima(AirPassengers, order=c(1,0,0), list(order=c(2,1,0), period=12))
fore <- predict(fit, n.ahead=24)
> # error bounds at 95% confidence level
U <- fore$pred + 2*fore$se
L <- fore$pred - 2*fore$se
#col=c(1,2,4,4)
#AirPassengers in black
#fore$pred in red
#Upper ND LOWER bound in blue 
ts.plot(AirPassengers, fore$pred, U, L, col=c(1,2,4,4), lty = c(1,1,2,2)) 
ts.plot(AirPassengers, fore$pred, U, L, col=c(1,2,4,4)) 
legend("topleft", c("Actual", "Forecast", "Error Bounds (95% Confidence)"),col=c(1,2,4), lty=c(1,1,2))


#8.4.1 Dynamic Time Warping
#Dynamic Time Warping (DTW) finds optimal alignment between two time series
#In that package, function dtw(x, y, ...) computes dynamic time warp and finds optimal alignment between two
#time series x and y, and dtwDist(mx, my=mx, ...) or dist(mx, my=mx, method="DTW", ...)
#calculates the distances between time series mx and my.
library(dtw)
idx <- seq(0, 2*pi, len=100)
a <- sin(idx) + runif(100)/10
b <- cos(idx)
align <- dtw(a, b, step=asymmetricP1, keep=T)
dtwPlotTwoWay(align)

#CH 9 - Association Rules
#########################
#9.2 The Titanic Dataset

str(Titanic)
df <- as.data.frame(Titanic)
head(df)
titanic.raw <- NULL
for(i in 1:4) {
  titanic.raw <- cbind(titanic.raw, rep(as.character(df[,i]), df$Freq))
  }
titanic.raw <- as.data.frame(titanic.raw)
names(titanic.raw) <- names(df)[1:4]
dim(titanic.raw)
str(titanic.raw)
summary(titanic.raw)


#Support, confidence and lift are three common measures for selecting interesting association
#rules. Besides them, there are many other interestingness measures, such as chi-square, conviction,
#gini and leverage


9.3 Association Rule Mining

> library(arules)
> # find association rules with default settings
  > rules.all <- apriori(titanic.raw)
> quality(rules.all) <- round(quality(rules.all), digits=3)
> rules.all
> inspect(rules.all)
> ## use code below if above code does not work
  > arules::inspect(rules.all)
> # rules with rhs containing "Survived" only
  > rules <- apriori(titanic.raw, control = list(verbose=F),
                     + parameter = list(minlen=2, supp=0.005, conf=0.8),
                     + appearance = list(rhs=c("Survived=No", "Survived=Yes"),
                                         + default="lhs"))
> quality(rules) <- round(quality(rules), digits=3)
> rules.sorted <- sort(rules, by="lift")
> inspect(rules.sorted)

#9.4 Removing Redundancy

# For example, the above rule 2 provides
# no extra knowledge in addition to rule 1, since rules 1 tells us that all 2nd-class children survived.
# Generally speaking, when a rule (such as rule 2) is a super rule of another rule (such as rule 1)
# and the former has the same or a lower lift, the former rule (rule 2) is considered to be redundant.
# Other redundant rules in the above result are rules 4, 7 and 8, compared respectively with rules
# 3, 6 and 5.

> # find redundant rules
  > subset.matrix <- is.subset(rules.sorted, rules.sorted)
> subset.matrix[lower.tri(subset.matrix, diag=T)] <- NA
> redundant <- colSums(subset.matrix, na.rm=T) >= 1
> which(redundant)
> # remove redundant rules
  > rules.pruned <- rules.sorted[!redundant]
> inspect(rules.pruned)

9.5 Interpreting Rules

# While it is easy to find high-lift rules from data, it is not an easy job to understand the identied
# rules. It is not uncommon that the association rules are misinterpreted to nd their business meanings.
# For instance, in the above rule list rules.pruned, the first rule "{Class=2nd, Age=Child}
# => {Survived=Yes}" has a confidence of one and a lift of three and there are no rules on children
# of the 1st or 3rd classes. Therefore, it might be interpreted by users as children of the 2nd
# class had a higher survival rate than other children. This is wrong! The rule states only that all
# children of class 2 survived, but provides no information at all to compare the survival rates of
# dierent classes. To investigate the above issue, we run the code below to nd rules whose rhs is
# "Survived=Yes" and lhs contains "Class=1st", "Class=2nd", "Class=3rd", "Age=Child" and
# "Age=Adult" only, and which contains no other items (default="none"). We use lower thresholds
# for both support and condence than before to nd all rules for children of dierent classes.

> rules <- apriori(titanic.raw,
                   + parameter = list(minlen=3, supp=0.002, conf=0.2),
                   + appearance = list(rhs=c("Survived=Yes"),
                                       + lhs=c("Class=1st", "Class=2nd", "Class=3rd",
                                               + "Age=Child", "Age=Adult"),
                                       + default="none"),
                   + control = list(verbose=F))
> rules.sorted <- sort(rules, by="confidence")
> inspect(rules.sorted)

9.6 Visualizing Association Rules
> library(arulesViz)
> plot(rules.all)
plot(rules.all, method="graph")
plot(rules.all, method="graph", control=list(type="items"))


plot(rules.all, method="paracoord", control=list(reorder=TRUE))



#CH 10 - Text Mining
####################
























