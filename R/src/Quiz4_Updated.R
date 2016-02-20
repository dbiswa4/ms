#Q-13

mol<-parse.smiles("c1([C@@H](C(O)=O)C)ccc(CC(C)C)cc1")[[1]]
get.xlogp(mol)

#Q-15
#use the parse.smiles again, then use length(get.atoms(mol)) to find the length of the vector of atoms
mol<-parse.smiles("C1CN(CCC1NC(C2=CN=CC=C2)C3=CN(N=C3)CC(F)(F)F)C4=CC=C(C=C4)C(F)(F)F")[[1]]
length(get.atoms(mol))

#Q-16
#Use the CDK Molecular Weight descriptor to find the molecular weight of the chemical compound with SMILES 
mol<-parse.smiles("N1(c2c(NC(c3c1nccc3)=O)c(ccn2)C)C1CC1")[[1]]
descName<-c("org.openscience.cdk.qsar.descriptors.molecular.WeightDescriptor")
desc <- eval.desc(mol, descName)

#Q-17
#What is the total number of bonds present in 
mol<-parse.smiles("C1CN(CCC1NC(C2=CN=CC=C2)C3=CN(N=C3)CC(F)(F)F)C4=CC=C(C=C4)C(F)(F)F")[[1]]
length(get.bonds(mol))

my.get.assay <- function(aid) {
  library(RCurl)
  library(jsonlite)
  urlcon <- sprintf('https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/%d/summary/JSON', as.integer(aid))
  h <- getCurlHandle()
  d <- getURL(urlcon, curl=h)
  j <- fromJSON(d)
  d<-j$AssaySummaries$AssaySummary
  return(j)
}
data=my.get.assay(363)

#Q-20
fp1 <- new("fingerprint", nbit=10, bits=c(1,2,4,6,9)) 
fp2 <- new("fingerprint", nbit=10, bits=c(2,5,6)) 
distance(fp1, fp2, "jaccard")
