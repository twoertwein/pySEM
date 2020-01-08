library('lavaan')

setwd('~/git/sem_python/tests')
data <- read.csv('HolzingerSwineford1939.csv')

model <- '
visual  =~ x1 + x2 + x3
textual =~ x4 + x5 + x6
speed   =~ x7 + x8 + x9
'

fit <- cfa(model, data = data, auto.fix.first = FALSE, std.lv = TRUE)
summary(fit)

