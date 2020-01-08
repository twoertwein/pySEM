library('lavaan')

setwd('~/git/sem_python/tests')
# data <- read.csv('PoliticalDemocracy_missing.csv')
data <- read.csv('PoliticalDemocracy.csv')

model <- '
  # measurement model
    ind60 =~ x1 + x2 + x3
    dem60 =~ y1 + y2 + y3 + y4
    dem65 =~ y5 + y6 + y7 + y8
  # regressions
    dem60 ~ ind60
    dem65 ~ ind60 + dem60
 # residual correlations
    y1 ~~ y5
    y2 ~~ y4 + y6
    y3 ~~ y7
    y4 ~~ y8
    y6 ~~ y8
'

fit <- sem(model, data = data, auto.fix.first = FALSE, std.lv = TRUE, missing = 'fiml')
summary(fit)
