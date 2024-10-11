library(tidyverse)

path = 'results/metrics/'
files = list.files(path)

model = lr = decay = iteration = acc = mse = runtime = c()
for (file in files){
  split = strsplit(file, '_')[[1]]

  model = c(model, split[1])
  lr = c(lr, split[2])
  decay = c(decay, split[3])
  iteration = c(iteration, as.numeric(substr(split[4], 1, nchar(split[4])-4)))
  
  res = read.table(paste0(path, file))
  acc = c(acc, res[1,1])
  mse = c(mse, res[2,1])
  runtime = c(runtime, res[3,1])
  
}

results = data.frame(model, lr, decay, iteration, acc, mse, runtime) 

results %>%
  group_by(model, lr, decay) %>%
  summarise(mean(acc), mean(mse), mean(runtime))

results %>%
  group_by(model, nclass) %>%
  summarise(se_acc=sd(acc)/sqrt(n()), 
            se_mse=sd(mse)/sqrt(n()), 
            acc=mean(acc), 
            mse=mean(mse)) %>%
  ggplot(aes(x=as.factor(nclass), y=acc, group=model, col=model)) +
    geom_line() +
    geom_ribbon(aes(ymin=acc-1.96*se_acc, ymax=acc+1.96*se_acc), alpha=.2, color=NA)



results %>%
  group_by(model, nclass) %>%
  summarise(se_acc=sd(acc)/sqrt(n()), 
            se_mse=sd(mse)/sqrt(n()), 
            acc=mean(acc), 
            mse=mean(mse)) %>%
  ggplot(aes(x=as.factor(nclass), y=mse, group=model, col=model)) +
  geom_line() +
  geom_ribbon(aes(ymin=mse-1.96*se_mse, ymax=mse+1.96*se_mse), alpha=.2, color=NA)
