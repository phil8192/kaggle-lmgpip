rm(list=ls())

# kaggle's gini function (ripped verbatim from https://www.kaggle.com/wiki/RCodeForGini)
#"NormalizedGini" is the other half of the metric. This function does most of the work, though
SumModelGini <- function(solution, submission) {
  df = data.frame(solution = solution, submission = submission)
  df <- df[order(df$submission, decreasing = TRUE),]
  df
  df$random = (1:nrow(df))/nrow(df)
  df
  totalPos <- sum(df$solution)
  df$cumPosFound <- cumsum(df$solution) # this will store the cumulative number of positive examples found (used for computing "Model Lorentz")
  df$Lorentz <- df$cumPosFound / totalPos # this will store the cumulative proportion of positive examples found ("Model Lorentz")
  df$Gini <- df$Lorentz - df$random # will store Lorentz minus random
  #print(df)
  return(sum(df$Gini))
}

NormalizedGini <- function(solution, submission) {
  SumModelGini(solution, submission) / SumModelGini(solution, solution)
}

# my auxiliary.

# all(range(sapply(1:69, function(x) norml.range(x, 0.1, 0.9, 1, 69))) == c(0.1, 0.9))
norml.range <- function(v, from, to, minv=min(v), maxv=max(v)) (((to-from)*(v-minv))/(maxv-minv))+from
# 0 == sum(abs(round(sapply(1:69, function(x) norml.range.rev(norml.range(x, 0.1, 0.9, 1, 69), 1, 69, 0.1, 0.9)), 1) - 1:69))
norml.range.rev <- function(v, minv, maxv, from=min(v), to=max(v)) (((v-from)*(maxv-minv))/(to-from))+minv
# mean 0, sd 1.
standardise <- function(v, m=mean(v), s=sd(v)) (v-m)/s
rmse <- function(des, out) sqrt(mean((des-out)^2))
mae <- function(des, out) mean(abs(des-out))

# conf.
valid.ratio <- 0

### prep data.
# training
train <- read.csv("data/train.csv", header=T)

# testing (used at end for final predictions)
# add dummy Hazard column as to match training data dim.
test.data <- read.csv("data/test.csv", header=T)
test.data <- cbind(Id=test.data[, 1], Hazard=-1, test.data[, -1])

# temp: add testing data to training data so that min,max values etc are
# the same in training and in the final prediction for normalisation.
train <- rbind(test.data, train)

# remove columns of apparent no information value (see random forest graph)
# T2_V10, T2_V7, T1_V13, T1_V10
train <- train[, -which(colnames(train) %in% c("T2_V10", "T2_V7", "T1_V13", "T1_V10"))]

# T1_V1: uniformish. 1:19. cant be a count of something: would expect more at 1, less at 19.
# could be randomised (hense uniform appearance). normalise for now. could try seperating into
# boolean variables or re-ordering by sum (based on assumption that data has been randomised.)
train$T1_V1 <- norml.range(train$T1_V1, -3, 3)

# T1_V2: 1:24, same as ^ ;although seems to be a bias towards 7 and 18.
train$T1_V2 <- norml.range(train$T1_V2, -3, 3)

# T1_V3: 1:9, distribution is apparent: exponential? decay from 1:9.
train$T1_V3 <- norml.range(train$T1_V3, -3, 3)

# T1_V4: 8 factors: B     C     E     G     H     N     S     W
# N is most frequent. N E S W, north, east, south, west?
# re-ordering by frequency or creating new boolean features may help..
# normalise for now.
train$T1_V4 <- norml.range(as.integer(train$T1_V4), -3, 3)

# T1_V5: 10 factors: A     B     C     D     E     H     I     J     K     L
# normalise for now.
train$T1_V5 <- norml.range(as.integer(train$T1_V5), -3, 3)

# T1_V6: 2 factors: Y N
# yes or no.
# no slightly more popular.
# normalise.
train$T1_V6 <- norml.range(as.integer(train$T1_V6), -3, 3)

# T1_V7: 4 factors: A     B     C     D
# normalise
train$T1_V7 <- norml.range(as.integer(train$T1_V7), -3, 3)

# T1_V8: 4 factors: A     B     C     D
# normalise
train$T1_V8 <- norml.range(as.integer(train$T1_V8), -3, 3)

# T1_V9: 6 factors: B     C     D     E     F     G
# normalise
train$T1_V9 <- norml.range(as.integer(train$T1_V9), -3, 3)

# T1_V10: 2:12.
# missing 4,5,6 and 9,10,11
# normalise.
#train$T1_V10 <- norml.range(train$T1_V10, 0.1, 0.9)

# T1_V11: 12 factors:
#    A     B     D     E     F     H     I     J     K     L     M     N 
# 1556 17047   258   450   544 15381  1364  6197   239  7003   541   419 
# normalise
train$T1_V11 <- norml.range(as.integer(train$T1_V11), -3, 3)

# T1_V12, 4 factors:
#    A     B     C     D 
# 1130 46900  1395  1574 
train$T1_V12 <- norml.range(as.integer(train$T1_V12), -3, 3)

# T1_V13. 5,10,15 or 20.
#train$T1_V13 <- norml.range(train$T1_V13, 0.1, 0.9)

# T1_V14, 0,1,2,3,4 (hardly any 0s)
train$T1_V14 <- norml.range(train$T1_V14, -3, 3)

# T1_V15:
#    A     C     D     F     H     N     S     W 
# 45680  1652   758    85   524  1879   191   230 
train$T1_V15 <- norml.range(as.integer(train$T1_V15), -3, 3)

# T1_V16:
#   A    B    C    D    E    F    G    H    I    J    K    L    M    N    O    P    Q    R 
# 2705 8933  808 1397 2599  187  484  408 9331 2410 8159  729 1264 2277  152  459  358 8339 
train$T1_V16 <- norml.range(as.integer(train$T1_V16), -3, 3)

# T1_V17:
#    N     Y 
# 41183  9816 
train$T1_V17 <- norml.range(as.integer(train$T1_V17), -3, 3)

# T2_V1 (1:100) ascending distribution. house age?
train$T2_V1 <- norml.range(train$T2_V1, -3, 3)

# T2_V2 (1:39) distribution. peaks at 8, then exponential decay toward 39.
train$T2_V2 <- norml.range(train$T2_V2, -3, 3)

# T2_V3
#     N     Y 
# 34548 16451
train$T2_V3 <- norml.range(as.integer(train$T2_V3), -3, 3)

# T2_V4
# distribution with peaks at 5 and 12
# 1:22
train$T2_V4 <- norml.range(train$T2_V4, -3, 3)

# T2_V5
#     A     B     C     D     E     F 
# 33845 11201  5013   515   412    13 
train$T2_V5 <- norml.range(as.integer(train$T2_V5), -3, 3)

# T2_V6
# exponential distribution peaking at 2, decaying to 7
# 1:7
train$T2_V6 <- norml.range(train$T2_V6, -3, 3)

# T2_V7
# 7 possible values:
# 22+cumsum(rep(3,7))-3
# 22 25 28 31 34 37 40.
# ascending distribution.
#train$T2_V7 <- norml.range(train$T2_V7, 0.1, 0.9)

# T2_V8 1,2,3
train$T2_V8 <- norml.range(train$T2_V8, -3, 3)

# T2_V9 distribution 1:25
train$T2_V9 <- norml.range(train$T2_V9, -3, 3)

# T2_V10 distribution 1:7
#train$T2_V10 <- norml.range(train$T2_V10, 0.1, 0.9)

# T2_V11 yes,no
train$T2_V11 <- norml.range(as.integer(train$T2_V11), -3, 3)

# T2_V12 yes,no
train$T2_V12 <- norml.range(as.integer(train$T2_V12), -3, 3)

# T2_V13
#     A     B     C     D     E 
# 10260   514  7507  5084 27634 
train$T2_V13 <- norml.range(as.integer(train$T2_V13), -3, 3)

# T2_V14 distribution 1:7
# peak at 2, decay toward 7.
train$T2_V14 <- norml.range(train$T2_V14, -3, 3)

# T2_V15 distribtion. looks cyclic. 1:12 (months?)
# peak at 1.
train$T2_V15 <- norml.range(train$T2_V15, -3, 3)

# output/target
#train$Hazard <- train$Hazard/100
# ^ squashes close to 0 (0.01), best normalise within [0.1, 0.9]
# (assume min and max value of [1, 100])
train$Hazard <- norml.range(train$Hazard, from=0.1, to=0.9, minv=1, maxv=100) 

# extract the testing data (can also id it by negative Hazard)
test.data <- head(train, nrow(test.data))
train <- tail(train, -nrow(test.data)) 

# randomise order of training data
train <- train[sample(1:nrow(train)), ]

# hold 10% of data back for validation (not used in training or model testing)
if(valid.ratio > 0) {
  valid <- head(train, as.integer(nrow(train)*valid.ratio)) 
  train <- tail(train, nrow(train)-as.integer(nrow(train)*valid.ratio))
}

# spit it out.
write.table(train[, c(3:ncol(train), 2)], "/tmp/train_data.csv", quote=F, row.names=F, col.names=F, sep=",")

if(valid.ratio > 0)
  write.table(valid[, c(3:ncol(valid), 2)], "/tmp/valid_data.csv", quote=F, row.names=F, col.names=F, sep=",")

write.table(test.data[, c(3:ncol(test.data), 2)], "/tmp/test_data.csv", quote=F, row.names=F, col.names=F, sep=",")


######### external: use neural-network-light

load.model <- function() {
  # look at errors
  errors <<- read.csv("/tmp/errors.csv", header=F)
  par(mfrow=c(1, 2))
  plot(errors[, 1], type="s", main="training error", xlab="epoch", ylab="error")
  plot(errors[, 2], type="s", main="testing error", xlab="epoch", ylab="error")

  ### look at model output
  # training error/fit
  train.out <<- read.csv("/tmp/train_out.csv", header=F)
  train.out <<- norml.range.rev(train.out[, 1], 1, 100, 0.1, 0.9) # 100*train.out[, 1]
  train.des <<- as.integer(round(norml.range.rev(train$Hazard, 1, 100, 0.1, 0.9), 1)) # as.integer(100*train$Hazard)

  if(valid.ratio > 0) {
    valid.out <<- read.csv("/tmp/valid_out.csv", header=F)
    valid.out <<- norml.range.rev(valid.out[, 1], 1, 100, 0.1, 0.9) # 100*valid.out[, 1]
    valid.des <<- as.integer(round(norml.range.rev(valid$Hazard, 1, 100, 0.1, 0.9), 1)) # as.integer(100*valid$Hazard)
  }
}
    
check.fit <- function() {
  print(paste("mae|rmse|gini training   =", round(mae(train.out, train.des), 6),
                                            round(rmse(train.out, train.des), 6),
                                            round(NormalizedGini(train.des, train.out), 6)))
  if(valid.ratio > 0) {
    # validation + check gini to give an expectation of kaggle rank
    print(paste("mae|rmse|gini validation =", round(mae(valid.out, valid.des), 6),
                                              round(rmse(valid.out, valid.des), 6),
                                              round(NormalizedGini(valid.des, valid.out), 6)))
  }
}

kaggle <- function() {
  #### make kaggle predictions
  test.out <- read.csv("/tmp/test_out.csv", header=F)
  test.out <- cbind(Id=test.data$Id, Hazard=norml.range.rev(test.out[, 1], 1, 100, 0.1, 0.9)) # 100*test.out[, 1])
  write.table(test.out, "/tmp/kaggle_out.csv", quote=F, row.names=F, col.names=T, sep=",")
}

# for single model
if(F) {
  load.model()
  check.fit()
}

# for k-fold models

load.models.k <- function(k) {
  train.dat        <<- NULL
  valid.dat        <<- NULL
  test.dat         <<- NULL
  errors.train.dat <<- NULL
  errors.test.dat  <<- NULL
  for(i in 1:k) {
    train.csv         <- read.csv(paste0("/tmp/train_out_", sprintf("%02d", i), ".csv"), header=F)
    valid.csv         <- read.csv(paste0("/tmp/valid_out_", sprintf("%02d", i), ".csv"), header=F)
    test.csv          <- read.csv(paste0("/tmp/test_out_", sprintf("%02d", i), ".csv"), header=F)
    errors.csv        <- read.csv(paste0("/tmp/errors_", sprintf("%02d", i), ".csv"), header=F)    
    train.dat        <<- rbind(train.dat, t(train.csv))
    valid.dat        <<- rbind(valid.dat, t(valid.csv))
    test.dat         <<- rbind(test.dat, t(test.csv))
    errors.train.dat <<- rbind(errors.train.dat, t(errors.csv[, 1]))
    errors.test.dat  <<- rbind(errors.test.dat, t(errors.csv[, 2]))
  }
  train.dat        <<- as.matrix(t(train.dat))
  valid.dat        <<- as.matrix(t(valid.dat))
  test.dat         <<- as.matrix(t(test.dat))
  errors.train.dat <<- as.matrix(t(errors.train.dat))
  errors.test.dat  <<- as.matrix(t(errors.test.dat))
}

# ensemble
if(F) {
  ve <- norml.range.rev(rowMeans(valid.dat), 1, 100, 0.1, 0.9)
  te <- cbind(Id=test.data$Id, Hazard=norml.range.rev(rowMeans(test.dat), 1, 100, 0.1, 0.9))

  valid.des <- as.integer(round(norml.range.rev(valid$Hazard, 1, 100, 0.1, 0.9), 1))
  print(paste("gini =", round(NormalizedGini(valid.des, ve), 6)))

  write.table(te, "/tmp/kaggle_ensemble.csv", quote=F, row.names=F, col.names=T, sep=",")
}

if(F) {
# check structure generalisation
x <- read.csv("/tmp/search.csv", header=F)
print(paste("optimal hidden nodes =", which.min(tapply(x[, 3], x[, 1], mean))))
# 4 0.03229
# 6 0.03237
# 8 0.03243
}



test.out <- read.csv("/tmp/test_out_32.csv", header=F)
test.out <- cbind(Id=test.data$Id, Hazard=norml.range.rev(test.out[, 1], 1, 100, 0.1, 0.9)) # 100*test.out[, 1])
write.table(test.out, "/tmp/kaggle_out_32.csv", quote=F, row.names=F, col.names=T, sep=",")

test.out <- read.csv("/tmp/test_out_27.csv", header=F)
test.out <- cbind(Id=test.data$Id, Hazard=norml.range.rev(test.out[, 1], 1, 100, 0.1, 0.9)) # 100*test.out[, 1])
write.table(test.out, "/tmp/kaggle_out_27.csv", quote=F, row.names=F, col.names=T, sep=",")

test.out <- read.csv("/tmp/test_out_19.csv", header=F)
test.out <- cbind(Id=test.data$Id, Hazard=norml.range.rev(test.out[, 1], 1, 100, 0.1, 0.9)) # 100*test.out[, 1])
write.table(test.out, "/tmp/kaggle_out_19.csv", quote=F, row.names=F, col.names=T, sep=",")

test.out <- read.csv("/tmp/test_out_22.csv", header=F)
test.out <- cbind(Id=test.data$Id, Hazard=norml.range.rev(test.out[, 1], 1, 100, 0.1, 0.9)) # 100*test.out[, 1])
write.table(test.out, "/tmp/kaggle_out_22.csv", quote=F, row.names=F, col.names=T, sep=",")

