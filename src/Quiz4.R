#Drug Discovery - Quiz 4

mol<-parse.smiles("C1CN(CCC1NC(C2=CN=CC=C2)C3=CN(N=C3)CC(F)(F)F)C4=CC=C(C=C4)C(F)(F)F")[[1]]
get.xlogp(mol)
length(get.atoms(mol))
descName<-c("org.openscience.cdk.qsar.descriptors.molecular.WeightDescriptor")
desc <- eval.desc(mol, descName)
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


fp1 <- new("fingerprint", nbit=10, bits=c(1,2,4,6,9)) 
fp2 <- new("fingerprint", nbit=10, bits=c(2,5,6)) 

distance(fp1, fp2, "jaccard")
