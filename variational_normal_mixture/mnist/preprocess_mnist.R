#http://stackoverflow.com/questions/21521571/how-to-read-mnist-database-in-r
library(Matrix)
library(RUnit)

kRepoLocation <- Sys.getenv("GIT_REPO_LOC")
setwd(file.path(kRepoLocation, "variational_bayes/variational_normal_mixture/mnist/"))


kSaveData <- TRUE

image.filename <- "train-images-idx3-ubyte"
label.filename <- "train-labels-idx1-ubyte" 

test.image.filename <- "t10k-images-idx3-ubyte"
test.label.filename <- "t10k-labels-idx1-ubyte"  

keep.components <- 15


#########

# Get summary statistics for the PCA
mnist.covariance.file <- "mnist_covariance_data.Rdata"
if (!file.exists(mnist.covariance.file)) {
  # Read the headers
  input.file <- file(image.filename, "rb")
  checkEquals(2051, readBin(input.file, integer(), n=1, endian="big"))
  n.images <- readBin(input.file, integer(), n=1, endian="big")
  n.rows <- readBin(input.file, integer(), n=1, endian="big")
  checkEquals(n.rows, readBin(input.file, integer(), n=1, endian="big"))
  
  pb <- txtProgressBar(0, n.images, style=3)
  
  vec.size <- n.rows * n.rows
  m.sum <- rep(0, vec.size)
  m.sum.squared <- matrix(0, vec.size, vec.size)
  count <- 0
  
  while (count < n.images) {
    setTxtProgressBar(pb, count)
    m <- readBin(input.file, integer(), size=1, n=vec.size, endian="big")
    m.sum <- m.sum + m
    m.sum.squared <- m.sum.squared + m %*% t(m)
    count <- count + 1
  }
  close(pb)
  close(input.file)
  
  if (kSaveData) {
    save(m.sum, m.sum.squared, n.images, n.rows, vec.size, file=mnist.covariance.file)
  }
} else {
  load(mnist.covariance.file)  
}

##################
# Get the means and eigenspace

m.mean <- m.sum / n.images
m.cov <- m.sum.squared / n.images - m.mean %*% t(m.mean)

cov.eigen <- eigen(m.cov)
#plot(cumsum(cov.eigen$values) / sum(cov.eigen$values)); abline(v=keep.components)

proj.mat <- cov.eigen$vectors[, 1:keep.components]
feat.mat <- matrix(NA, n.images, keep.components)

#################
# Read the low-dimensional training features into a matrix.

pb <- txtProgressBar(0, n.images, style = 3)
count <- 0

# Read the headers
input.file <- file(image.filename, "rb")
checkEquals(2051, readBin(input.file, integer(), n=1, endian="big"))
checkEquals(n.images, readBin(input.file, integer(), n=1, endian="big"))
checkEquals(n.rows, readBin(input.file, integer(), n=1, endian="big"))
checkEquals(n.rows, readBin(input.file, integer(), n=1, endian="big"))
while (count < n.images) {
  setTxtProgressBar(pb, count)
  m <- readBin(input.file, integer(), size=1, n=vec.size, endian="big")
  count <- count + 1
  feat.mat[count, ] <- (m - m.mean) %*% proj.mat
}
close(pb)
close(input.file)


#################
# Read a small number of raw images for sanity checking

raw.images <- list()
n.raw.images <- 100
pb <- txtProgressBar(0, n.raw.images, style = 3)
count <- 0

# Read the headers
input.file <- file(image.filename, "rb")
checkEquals(2051, readBin(input.file, integer(), n=1, endian="big"))
checkEquals(n.images, readBin(input.file, integer(), n=1, endian="big"))
checkEquals(n.rows, readBin(input.file, integer(), n=1, endian="big"))
checkEquals(n.rows, readBin(input.file, integer(), n=1, endian="big"))
while (count < n.raw.images) {
  setTxtProgressBar(pb, count)
  m <- readBin(input.file, integer(), size=1, n=vec.size, endian="big")
  count <- count + 1
  raw.images[[count]] <- matrix(m, n.rows, n.rows)
}
close(pb)
close(input.file)


#################
# Read the training labels.

labels <- rep(0, n.images)
pb <- txtProgressBar(0, n.images, style = 3)
count <- 0

# Read the headers
input.file <- file(label.filename, "rb")
checkEquals(2049, readBin(input.file, integer(), n=1, endian="big"))
checkEquals(n.images, readBin(input.file, integer(), n=1, endian="big"))
while (count < n.images) {
  setTxtProgressBar(pb, count)
  count <- count + 1
  labels[count] <- readBin(input.file, integer(), size=1, n=1, endian="big")
}
close(pb)
close(input.file)

# Sanity check
#index <- 10
#image(raw.images[[index]], main=labels[index])




#################
# Read the low-dimensional testing features into a matrix.
count <- 0

# Read the headers
input.file <- file(test.image.filename, "rb")
checkEquals(2051, readBin(input.file, integer(), n=1, endian="big"))
n.test.images <- readBin(input.file, integer(), n=1, endian="big")
checkEquals(n.rows, readBin(input.file, integer(), n=1, endian="big"))
checkEquals(n.rows, readBin(input.file, integer(), n=1, endian="big"))
test.feat.mat <- matrix(NA, n.test.images, keep.components)
pb <- txtProgressBar(0, n.test.images, style = 3)
while (count < n.test.images) {
  setTxtProgressBar(pb, count)
  m <- readBin(input.file, integer(), size=1, n=vec.size, endian="big")
  count <- count + 1
  test.feat.mat[count, ] <- (m - m.mean) %*% proj.mat
}
close(pb)
close(input.file)



#################
# Read the test labels.
test.labels <- rep(0, n.test.images)
pb <- txtProgressBar(0, n.test.images, style = 3)
count <- 0

# Read the headers
input.file <- file(test.label.filename, "rb")
checkEquals(2049, readBin(input.file, integer(), n=1, endian="big"))
checkEquals(n.test.images, readBin(input.file, integer(), n=1, endian="big"))
while (count < n.test.images) {
  setTxtProgressBar(pb, count)
  count <- count + 1
  test.labels[count] <- readBin(input.file, integer(), size=1, n=1, endian="big")
}
close(pb)
close(input.file)



################
# Save everything
training.filename <- sprintf("full_mnist_%d_features.Rdata", keep.components)
if (kSaveData) {
  save(feat.mat, labels,
       test.feat.mat, test.labels,
       raw.images,
       m.sum, m.sum.squared, count, proj.mat,
       file=training.filename)  
}




