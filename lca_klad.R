# Set seed for reproducibility
set.seed(iteration)

# Generate class probabilities
class_probs <- rep(1/nclass, nclass)

# Generate conditional probabilities
cond_probs <- matrix(0.5, nclass, nitems)
for (i in 1:nclass) {
  num_ones <- sample(50:100, 1)
  indices <- sample(nitems, num_ones, replace = FALSE)
  cond_probs[i, indices] <- 0.7
}

# Generate true class membership for each person
true_class_ix <- sample(1:nclass, N, replace = TRUE, prob = class_probs)
true_class <- matrix(0, N, nclass)
true_class[cbind(1:N, true_class_ix)] <- 1

# Simulate responses
prob <- true_class %*% cond_probs
data <- matrix(rbinom(N * nclass, 1, prob), N, nclass)


