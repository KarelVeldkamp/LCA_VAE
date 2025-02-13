library(poLCA)
library(clue)

args = commandArgs(trailingOnly = TRUE)


N = 10000
for (NCLASS in c(2,4,8)){
  for (NREP in c(1, 5)){
    for(NITEMS in c(100)){
      for (iteration in 1:20){
        # read data and true parameters
        data = read.csv(paste0(c('~/Documents/GitHub/LCA_VAE/true/data/data_', NCLASS, '_', N, '_', iteration, '_', NITEMS, '.csv'), collapse = ''), header = F) +1
        true_class = read.csv(paste0(c('~/Documents/GitHub/LCA_VAE/true/parameters/class_',NCLASS, '_', N, '_', iteration, '_', NITEMS,'.csv'), collapse = ''), header=F)
        true_probs = as.matrix(read.csv(paste0(c('~/Documents/GitHub/LCA_VAE/true/parameters/probs_',NCLASS, '_', N, '_', iteration, '_', NITEMS,'.csv'), collapse = ''), header=F))
        
        # fit model
        f <- as.formula(paste("cbind(", paste(colnames(data), collapse = ","), ") ~ 1"))
        t1 = Sys.time()
        lca = poLCA(f, data, nclass = NCLASS, nrep = NREP)
        runtime = as.numeric(Sys.time()-t1,units="secs")
        
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
          est = as.matrix(estimates[[i]])
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
  
        print(paste0(c('~/Documents/GitHub/LCA_VAE/results/estimates/est_lca_', NCLASS, '_', N, '_', iteration, '_', NITEMS, '_', NREP, '.csv'), collapse=''))
        write.csv(results, paste0(c('~/Documents/GitHub/LCA_VAE/results/estimates/est_lca_', NCLASS, '_', N, '_', iteration, '_', NITEMS, '_', NREP, '.csv'), collapse=''))
        
        # write metrics to file
  
        fileConn<-file(paste0(c('~/Documents/GitHub/LCA_VAE/results/metrics/lca_', NCLASS, '_', N, '_', iteration, '_', NITEMS, '_', NREP, '.txt'), collapse=''))
        writeLines(c(as.character(acc), as.character(mse), as.character(runtime)),fileConn)
        close(fileConn)
        
      }
    }
  }
}






sample_theta <-function(probs){
  selected_indices <- apply(probs, 1, function(row_probs) sample.int(length(row_probs), 1, prob = row_probs))
  latent_samples = matrix(0, nrow(probs), ncol(probs))
  latent_samples[cbind(1:10000, selected_indices)] =1 
  return(latent_samples)
}



elbo <- function(theta, probs, data){
  log_ratio = log(pmax(theta*ncol(theta), 1e-7))
  kl = apply(theta * log_ratio, 1, sum)
  
  sample = sample_theta(theta)
  
  y_hat = as.matrix(sample %*% probs)
  y = as.matrix(data-1)
  
  lll = log(pmax(y * y_hat, 1e-7))+ log(pmax((1-y) * (1-y_hat), 1e-7))
  lll = apply(lll, 1, sum)
  #bce = mean(lll) #* ncol(lll)
  
  
  
  return(mean(-lll + kl))
}


probs = do.call(cbind,lapply(lca$probs, function(x) x[,1]))
load('~/Documents/GitHub/LCA_VAE/tmp/lca.RData')
vae_posterior = as.matrix(read.csv('~/Documents/GitHub/LCA_VAE/tmp/pred_class.csv', header = F))
vae_probs = as.matrix(read.csv('~/Documents/GitHub/LCA_VAE/tmp/est_probs.csv', header=F))

elbo(vae_posterior, vae_probs, data)
elbo(lca$posterior, probs, data)


posterior =vae_posterior
probs = vae_probs
profile_plot <- function(class_ix, item_ix){
  elbos = c()
  ps = seq(probs[class_ix,item_ix]-.05,probs[class_ix,item_ix]+.05,length.out=30)
  for (p in ps){
    new_probs = probs
    new_probs[class_ix,item_ix] = p
    elbos = c(elbos, elbo(posterior, new_probs, data))
  }
  
  plot(ps, elbos, type='l')
  abline(v=probs[class_ix,item_ix])
}

profile_plot(10,108)
