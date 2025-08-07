####### Data Generating Process #######
generate_data_AA <- function(p, pos, u, seed){
  # Ordering the columns with decreasing abundance
  dcount <- count[, order(decreasing = TRUE, colSums(count, na.rm = TRUE), 
                          apply(count, 2L, paste, collapse = ''))] 
  
  ## Randomly sampling patients from 574 observations
  set.seed(seed)
  norm_count <- count / rowSums(count)
  col_means <- colMeans(norm_count > 0)
  indices <- which(col_means > 0.2)
  sorted_indices <- indices[order(col_means[indices], decreasing = TRUE)]
  
  dcount <- count[, sorted_indices][, 1:p]
  sel_index <- sort(sample(1:nrow(dcount), 500))
  dcount <- dcount[sel_index, ]
  original_OTU <- dcount + 0.5
  seq_depths <- rowSums(original_OTU)
  Pi <- sweep(original_OTU, 1, seq_depths, "/")
  n <- nrow(Pi)
  
  ## Generating binary responses (case=1, control=0)
  Y <- sample(rep(0:1, each=250)) 
  
  # Randomly select 30 biomarkers from top 200 abundant features
  biomarker_idx <- sample(1:200, 30, replace = FALSE)
  # Assign effect directions
  n_positive <- round(30 * pos / 100)
  n_negative <- 30 - n_positive
  signs <- sample(rep(c(1, -1), c(n_positive, n_negative)))
  
  # Modify proportions for selected biomarkers
  Delta <- 3 # could be adjusted to have enough power
  fold_changes <- runif(30, min = 0, max = Delta)
  Pi_new <- Pi
  for (j in 1:30) {
    col_j <- biomarker_idx[j]
    if (signs[j] == 1) {
      Pi_new[Y == 1, col_j] <- Pi[Y == 1, col_j] * (1 + fold_changes[j])
    } else {
      Pi_new[Y == 0, col_j] <- Pi[Y == 0, col_j] * (1 + fold_changes[j])
    }
  }
  Pi_new <- Pi_new / rowSums(Pi_new) # renormalize
  
  # Draw sequencing depths for current samples
  template_seq_depths <- rowSums(count)
  # The sequencing depth was randomly drawn from the pool of sequencing depth of the template data.
  drawn_depths <- sample(template_seq_depths, size = n, replace = TRUE)
  adjusted_depths <- drawn_depths
  adjusted_depths[Y == 1] <- drawn_depths[Y == 1] * (1 + u) # Apply depth increase to case group
  
  ## Generate simulated count data
  sim_count <- matrix(0, nrow = n, ncol = p)
  for (i in 1:n) {
    sim_count[i, ] <- rmultinom(1, size = adjusted_depths[i], prob = Pi_new[i, ])
  }
  colnames(sim_count) <- colnames(Pi)
  
  return(list(Y = Y, X = sim_count, signal_indices = biomarker_idx))
}
