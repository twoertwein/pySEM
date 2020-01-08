library('lavaan')

setwd('~/git/sem_python/tests')
data <- read.csv('Demo_twolevel.csv')

model <- '
        level: 1
            fw =~ y1 + y2 + y3
            fw ~ x1 + x2 + x3
        level: 2
            fb =~ y1 + y2 + y3
            fb ~ w1 + w2
'

fit <- sem(model = model, data = Demo.twolevel, cluster = "cluster", auto.fix.first = FALSE, std.lv = TRUE)
summary(fit)
