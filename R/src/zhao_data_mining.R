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
//install.packages("rgl")
library(rgl)
plot3d(iris$Petal.Width, iris$Sepal.Length, iris$Sepal.Width)


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















