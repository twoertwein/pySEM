library('OpenMx')
 
setwd('~/git/sem_python/tests')
data <- read.csv('Demo_twolevel_missing.csv')
#data <- read.csv('Demo_twolevel.csv')
 
y <- c('y1', 'y2', 'y3')
x <- c('x1', 'x2', 'x3')
w <- c('w1', 'w2')
manifest_l1 <- c(y)
latent_l1 <- c('fw', x)
manifest_l2 <- c()
latent_l2 <- c('fb', y, w)
level2 <- data[!duplicated(data$cluster),]
 
between <- mxModel(
  "between",
  type="RAM",
  manifestVars=manifest_l2,
  latentVars=latent_l2,
  mxData(observed=level2, type="raw", primaryKey="cluster"),
  mxPath(from=y, arrows=2, free=TRUE, values=diag(var(data[y], na.rm=TRUE)), labels=paste0("y",1:3,"_var")),
  mxPath(from='fb', arrows=2, free=FALSE, values=1, labels="fb_var"),
  mxPath(from=y, arrows=2, free=TRUE, values=1, labels=paste0("y",1:3,"_var_b")),
  
  mxPath(from='fb', to=y, arrows=1, free=TRUE, values=1, labels=c('fb_y1', 'fb_y2', 'fb_y3')),
  
  mxPath(from=w, to='fb', free=TRUE, labels=paste0('w',1:2,"_fb")),
  mxPath(from='one', to=w, free=FALSE, labels=paste0("data.w",1:2)),
  mxPath(from='one', to=y, free=TRUE))
 
within <- mxModel(
  "Demo_twolevel",
  type="RAM",
  between,
  manifestVars=manifest_l1,
  latentVars=latent_l1,
  mxData(observed=data, type="raw"),
  mxPath(from=y, arrows=2, free=TRUE, values=diag(var(data[y], na.rm=TRUE)) / 2, labels=paste0("y",1:3,"_var_w")),
  mxPath(from='fw', arrows=2, free=FALSE, values=1, labels="fw_var"),
 
  mxPath(from='fw', to=y, arrows=1, free=TRUE, values=1, labels=c('fw_y1', 'fw_y2', 'fw_y3')),
  mxPath(from=x, to='fw', arrows=1, free=TRUE, values=1, labels=c('x1_fw', 'x2_fw', 'x3_fw')),

  mxPath(paste0('between.y',1:3), paste0('y',1:3), values=1, free=FALSE, joinKey="cluster"),
 
  mxPath(from='one', to=manifest_l1, values=0, free=FALSE),
  mxPath(from='one', to=x, free=FALSE, values=0, labels=paste0("data.x",1:3)))
 
fit <- mxRun(within)
 
summary(fit)

