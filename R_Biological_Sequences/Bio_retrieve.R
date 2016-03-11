# Using Bioconductor packages for Retrieving data.
# Analysing sequences and accessing sequnce data from Genbank and other resources
# You can get the sequin R vignette from http://seqinr.r-forge.r-project.org/seqinr_2_0-1.pdf

# Function to check bioconductor packages
installpkg <- function (pkg){
if (!require(pkg, character.only=T)){
source("http://bioconductor.org/biocLite.R")
biocLite(pkg)
}else{
require(pkg, character.only=T)
}
}

installpkg('seqinr')

library("ggplot2")

#Gives list of available banks
choosebank()

choosebank(infobank = TRUE)[1:4, ]
banknameSocket=choosebank("embl")
str(banknameSocket)
choosebank("genbank")

?query
completeCDS <- query("completeCDS", "sp=Arabidopsis thaliana AND t=cds AND NOT k=partial")
#bb <- query("bb", "sp=Borrelia burgdorferi")
attributes(completeCDS)
nseq <- completeCDS$nelem
nseq #Shows number of sequences

seq <- getSequence(completeCDS$req[[1]]) # Accessing the first sequence
seq[1:20]
basecount <- table(seq)
#Total Basecount = 495
myseqname <- getName(completeCDS$req[[1]])
myseqname
dotchart(basecount, xlim = c(0, max(basecount)), pch = 19,main=paste("Basecount of ",myseqname))

#Codon usage of the sequence from seqinR package
#Codon is the amino acid name like Serine, Leucine, etc.
codonusage <- uco(seq)
#Total Codon => 495/3 = 165
dotchart.uco(codonusage, main = paste("Codon usage in", myseqname))
#getTrans is generic function to translate coding sequences into protein. 
#There is one to one mapping between codon and proteing alphabet
aacount <- table(getTrans(getSequence(completeCDS$req[[1]])))
aacount <- aacount[order(aacount)]
aacount
names(aacount) <- aaa(names(aacount))
aacount
#renames the column to codon instead of protein alphabets
dotchart(aacount, pch = 19, xlab = "Stop and amino-acid counts",
main = "There is only one stop codon")
#v is the vertical parameter and lty is the dotted line parameter
abline(v = 1, lty = 5)

choosebank("swissprot")
leprae = query("leprae", "AC=Q9CD83")
leprae <- getSequence(leprae$req[[1]])
ulcerans = query("ulcerans", "AC=A0PQ23")
ulcerans <- getSequence(ulcerans$req[[1]])
closebank()
dotPlot(leprae, ulcerans)

#using local fasta file
setwd("C:\\Users\\Ankita\\Documents\\ms\\R_Biological_Sequences")
hae <- read.fasta(file = "test.fasta",seqtype="AA")
length(hae)
a1<-hae[[2]]
a1
taborder <- a1[order(a1)]
names(taborder) <- aaa(names(taborder))#Convert one letter to three letter one
dotchart(table(taborder),pch=19,xlab="amino-acid-counts")
abline(v=1,lty=2)

#Compute Isoelectric point
#Isoelectric point (pI) is a pH in which net charge of protein is zero
computePI(a1)

#Compute Molecular Weight
pmw(a1)

#Creating Hydropathy scores 
data(EXP)
names(EXP$KD) <- sapply(words(), function(x) translate(s2c(x)))
kdc <- EXP$KD[unique(names(EXP$KD))]
kdc
kdc <- -kdc[order(names(kdc))]

# Hydropathy plot
hydro <- function(data, coef) { #data are sequences
f <- function(x) {
freq <- table(factor(x, levels = names(coef)))/length(x)
return(coef %*% freq) }
res <- sapply(data, f)
names(res) <- NULL
return(res)
}
a<-hydro(a1,kdc)
aa<-aaa(a1)
dat<-data.frame(aa,a)

#Using ggplot to plot hydropathy plot
#A hydrophilicity plot is a quantitative analysis of the degree of hydrophobicity 
#or hydrophilicity of amino acids of a protein. It is used to characterize or identify 
#possible structure or domains of a protein.
library(ggplot2)
#Need to check from video
qplot(seq_along(dat$a), dat$a)+geom_line(colour="blue")+geom_hline(yintercept=1)

#Searchign for patterns in genbank
installpkg("Biostrings")
choosebank("genbank")
PKhs = query("PKhs","sp=homo sapiens AND k=PRKA@")
kinase <- sapply(PKhs$req, getSequence)
PKA<-c2s(kinase[[1]])
length(kinase)
pattern<- "tcg"
#Need to check from video
matchPattern(pattern, PKA, max.mismatch = 0) #Biostrings package
