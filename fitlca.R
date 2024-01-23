library(poLCA)
library(clue)

args = commandArgs(trailingOnly = TRUE)

NCLASS = args[1]
NREP = args[2]

# read data and true parameters
data = read.csv(paste0(c('LCA_VAE/true/data/data_', NCLASS, '_', NREP, '.csv'), collapse = ''), header = F) +1
true_class = read.csv(paste0(c('LCA_VAE/true/parameters/class_',NCLASS,'_',NREP,'.csv'), collapse = ''), header=F)
true_probs = as.matrix(read.csv(paste0(c('LCA_VAE/true/parameters/probs_',NCLASS,'_',NREP,'.csv'), collapse = ''), header=F))

# fit model
f <- as.formula(paste("cbind(", paste(colnames(data), collapse = ","), ") ~ 1"))
lca = poLCA(f, data, nclass = NCLASS, nrep = NREP)

# save estimated parameters
est_probs =  t(as.matrix(do.call(rbind, lapply(lca$probs, function(mat) mat[, 2]))))
est_class_probs = lca$posterior

#plot(as.vector(est_probs), as.vector(true_probs))

# assign classes to fix label switching
new_order = as.vector(clue::solve_LSAP(abs(cor(t(est_probs), t(true_probs))), maximum = T))
true_probs = true_probs[new_order, ]
true_class = true_class[, new_order]


est_class = apply(est_class_probs, 1, which.max)
true_class = apply(true_class, 1, which.max)

acc = mean(est_class == true_class)
mse = mean((true_probs - est_probs)^2)


par = c()
value = c()
par_i = c()
par_j = c()
estimates = list(est_class_probs, est_class)
par_names = c('conditional', 'class')
for (i in 1:2){
  est = estimates[[i]]
  for (r in 1:nrow(est)){
    for (c in 1:ncol(est)){
      par = c(par, par_names[i])
      par_i = c(par_i, r)
      par_j = c(par_j, c)
      value = c(value, est[r, c])
    }
  }
}

results = data.frame('model'='LCA',
                     'nclass'=NCLASS,
                     'n_rep'=NREP,
                     'iteration'=iteration,
                     'parameter'=par,
                     'i'=par_i,
                     'j'=par_j,
                     'value'=value)

# write estimates to file
write.csv(results, paste0('results/estimates/est_LCA_', NCLASS, '_', NREP, '.csv'), collapse='')

# write metrics to file
fileConn<-file(paste0('/results/metrics/', paste(args, collapse='_')))
writeLines(c(as.character(acc), as.character(mse)),fileConn)
close(fileConn)


