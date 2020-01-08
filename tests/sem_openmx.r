library('OpenMx')

setwd('~/git/sem_python/tests')
# data <- read.csv('PoliticalDemocracy_missing.csv')
data <- read.csv('PoliticalDemocracy.csv')

ind60 <- c('x1', 'x2', 'x3')
dem60 <- c('y1', 'y2', 'y3', 'y4')
dem65 <- c('y5', 'y6', 'y7', 'y8')
manifest <- c(ind60, dem60, dem65)
latent <- c('ind60', 'dem60', 'dem65')
data <- data[manifest]

model <- mxModel(
    "PoliticalDemocracy",
    type="RAM",
    manifestVars=manifest,
    latentVars=latent,
    mxData(observed=data, type="raw"),
    mxPath(from=manifest, arrows=2, free=TRUE, values=diag(var(data, na.rm=TRUE)) / 2),
    mxPath(from='y1', to='y5', arrows=2, free=TRUE, values=0),
    mxPath(from='y3', to='y7', arrows=2, free=TRUE, values=0),
    mxPath(from='y2', to=c('y4', 'y6'), arrows=2, free=TRUE, values=0),
    mxPath(from='y4', to='y6', arrows=2, free=TRUE, values=0),
    mxPath(from='y6', to='y8', arrows=2, free=TRUE, values=0),
    mxPath(from=latent, arrows=2, free=FALSE, values=1),

    mxPath(from='ind60', to='dem60', arrows=1, free=TRUE, values=1, labels='ind60_dem60'),
    mxPath(from='ind60', to='dem65', arrows=1, free=TRUE, values=1, labels='ind60_dem65'),
    mxPath(from='dem60', to='dem65', arrows=1, free=TRUE, values=1, labels='dem60_dem65'),

    mxPath(from="ind60", to=ind60, arrows=1, free=TRUE, values=1, labels=c('indo60_x1', 'ind60_x2', 'ind60_x3')),
    mxPath(from="dem60", to=dem60, arrows=1, free=TRUE, values=1, labels=c('dem60_y1', 'dem60_y2', 'dem60_y3', 'dem60_y4')),
    mxPath(from="dem65", to=dem65, arrows=1, free=TRUE, values=1, labels=c('dem65_y1', 'dem65_y2', 'dem65_y3', 'dem65_y4')),
    mxPath(from='one', to=manifest, free=TRUE, values=colMeans(data, na.rm=TRUE)))

fit <- mxRun(model)

summary(fit)
