library(devtools)
remotes::install_github("egenn/rtemis") #user prompted for input
library(rtemis)
packageVersion("rtemis")

parkinsons <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data")
parkinsons$Status <- factor(parkinsons$status, levels = c(1, 0))
parkinsons$status <- NULL
parkinsons$name <- NULL
checkData(parkinsons)

parkinsons.addtree <- s.ADDTREE(parkinsons, gamma = .8, learning.rate = .1)

mplot3.addtree(parkinsons.addtree)

parkinsons.addtree$mod$addtree.pruned

print(parkinsons.addtree$mod$addtree.pruned, "Estimate")

predict(parkinsons.addtree)

res <- resample(parkinsons, n.resamples = 10, resampler = "kfold", verbose = TRUE)

parkinsons.train <- parkinsons[res$Fold_1, ]
parkinsons.test <- parkinsons[-res$Fold_1, ]

parkinsons.addtree <- s.ADDTREE(parkinsons.train, x.test = parkinsons.test,
                                gamma = .8, learning.rate = .1)

parkinsons.addtree.tune <- s.ADDTREE(parkinsons.train, x.test = parkinsons.test,
                                     gamma = seq(.6, .9, .1), learning.rate = .1)

parkinsons.addtree.tune <- s.ADDTREE(parkinsons.train, x.test = parkinsons.test,
                                     gamma = seq(.6, .9, .1), learning.rate = .1,
                                     grid.resampler.rtset = rtset.resample(resampler = 'strat.boot',
                                                                           n.resamples = 5))

parkinsons.addtree.tune$extra$gridSearch$tune.results

parkinsons.addtree.10fold <- elevate(parkinsons, mod = 'addtree', 
                                     gamma = c(.8, .9), 
                                     learning.rate = c(.01, .05),
                                     seed = 2018)

parkinsons.addtree.10fold

