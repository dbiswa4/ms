#RDataMining-book.pdf
#
setwd("/Users/dbiswas/Documents/MS/ds_study/ms/")

#1.3.1 The Iris Dataset
str(iris)

library(mboost)
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
library(RODBC)
connection <- odbcConnect(dsn="servername",uid="userid",pwd="******")
query <- "SELECT * FROM lib.table WHERE ..."
# or read query from file
   # query <- readChar("data/myQuery.sql", nchars=99999)
   myData <- sqlQuery(connection, query, errors=TRUE)
 odbcClose(connection)

#There are also sqlSave() and sqlUpdate() for writing or updating a table in an ODBC database.
#While package RODBC can read and write EXCEL les on Windows, but it does not work
#directly on Mac OS X, because an ODBC driver for EXCEL is not provided by default on Mac.

#2.4.2 Output to and Input from EXCEL Files
 library(RODBC)
 filename <- "data/myExcelFile.xls"
 xlsFile <- odbcConnectExcel(filename, readOnly = FALSE)
 sqlSave(xlsFile, a, rownames = FALSE)
 b <- sqlFetch(xlsFile, "sheetname")
 odbcClose(xlsFile)

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




























