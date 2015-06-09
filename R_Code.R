newf <- read.csv(file = "u.data",sep = "\t")

View(newf)

str(newf)

hist(newf$X3)

R1 <- as.matrix(newf)

install.packages("NMF")
library(NMF)
set.seed(12345)

res <- nmf(R1,rank = 10,nrun = 20)
res
?nmf
V.hat <- fitted(res) 
print(V.hat)

w <- basis(res) 
dim(w) 
print(w) 

R1

h <- coef(res) 
dim(h) 
plot(h) 


