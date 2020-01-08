library('OpenMx')

setwd('~/git/sem_python/tests')
data <- read.csv('HolzingerSwineford1939.csv')

visual <- c('x1', 'x2', 'x3')
textual <- c('x4', 'x5', 'x6')
speed <- c('x7', 'x8', 'x9')
manifest <- c(visual, textual, speed)
latent <- c('visual', 'textual', 'speed')
data <- data[manifest]

model <- mxModel(
    "HolzingerSwineford1939",
    type="RAM",
    manifestVars=manifest,
    latentVars=latent,
    mxData(observed=data, type="raw"),
    mxPath(from=manifest, arrows=2, free=TRUE, values=diag(var(data)) / 2),
    mxPath(from=latent, arrows=2, free=FALSE, values=1),
    mxPath(from=latent, connect='unique.bivariate', arrows=2, free=TRUE, values=1),
    mxPath(from="visual",to=visual, arrows=1, free=TRUE, values=1, labels=c('visual_x1', 'visual_x2', 'visual_x3')),
    mxPath(from="textual",to=textual, arrows=1, free=TRUE, values=1, labels=c('textual_x4', 'textual_x5', 'textual_x6')),
    mxPath(from="speed",to=speed, arrows=1, free=TRUE, values=1, labels=c('speed_x7', 'speed_x8', 'speed_x9')),
    mxPath(from='one', to=manifest, free=FALSE, values=colMeans(data)))

fit <- mxRun(model)

summary(fit)
