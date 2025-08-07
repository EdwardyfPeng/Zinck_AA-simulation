# lda_complete_parallel.R

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("need 4 parameters: p pos u fdr")
}

p <- as.numeric(args[1])
pos <- as.numeric(args[2])
u <- as.numeric(args[3])
target_fdr <- as.numeric(args[4])

cat(sprintf("begin complete LDA-KF simulation: p=%d, pos=%d, u=%.1f, target_fdr=%.2f\n", p, pos, u, target_fdr))

load("count.Rdata")
library(rstan)
library(glmnet)
library(knockoff)    
library(topicmodels) 
library(randomForest)
library(tidyverse)
library(stats)
library(ranger)
library(kosel)
library(reshape2)
library(lubridate)
library(Matrix)
library(parallel)
library(foreach)
library(doParallel)

log_normalize <- function(X) {
  # Ensure input is a matrix
  if (!is.matrix(X)) {
    stop("Input must be a matrix.")
  }
  # Ensure all elements in the matrix are numeric
  if (!all(is.numeric(X))) {
    stop("Matrix contains non-numeric values.")
  }
  # Add a pseudo-count of 0.5 to zero values
  X[X == 0] <- 0.5
  # Make compositional by dividing each value by its row sum
  X <- sweep(X, 1, rowSums(X), FUN="/")
  # Apply log transformation
  Z <- log(X)
  return(Z)
}

generateKnockoff <- function(X, Theta, Beta, seed = NULL) {
  D=nrow(Theta); V=ncol(Beta); N=rowSums(X)
  if(V == ncol(X)){
    colnames(Beta) <- colnames(X)
  }else if(V < ncol(X)){
    # Get the names of columns that are missing in Beta
    missing_cols <- setdiff(colnames(X), colnames(Beta))
    
    # For each missing column, add a column of zeros to Beta
    for (col in missing_cols) {
      Beta <- cbind(Beta, 0)
      colnames(Beta)[ncol(Beta)] <- col
    }
    
    # Reorder the columns of Beta to match the order in X
    Beta <- Beta[, colnames(X)]
  }
  # generate 1 sample
  generateSample <- function(N, theta, beta) {
    
    sample <- vector(length = N)
    z_d <- vector(length = N)
    for (n in 1:N) {
      z_n <- rmultinom(1, 1, theta)
      w_n <- rmultinom(1, 1, beta[which(z_n == 1),])
      sample[n] <- colnames(beta)[which(w_n == 1)]
      z_d[n] <- which(z_n == 1)
      names(z_d)[n] <- sample[n]
    }
    return(list(sample = sample, z = z_d))
  }
  
  # generate n samples
  cohort <- vector(mode = "list", length = D)
  z <- vector(mode = "list", length = D)
  for (d in 1:D) {
    sample.d <- generateSample(N[d], Theta[d, ], Beta)
    cohort[[d]] <- sample.d[["sample"]]
    z[[d]] <- sample.d[["z"]]
  }
  
  # collapse list of vectors into list of count tables
  sampleTaxaFreq <- lapply(cohort, table)
  
  # count matrix
  x_tilde <- matrix(data = 0, nrow = D, ncol = ncol(X))
  #rownames(x_tilde) <- rownames(Theta)
  colnames(x_tilde) <- colnames(Beta)
  
  for (d in 1:D) {
    x_tilde[d, names(sampleTaxaFreq[[d]])] <- sampleTaxaFreq[[d]]
  }
  
  return(x_tilde)
}

zinck.filter <- function(X, X_tilde, Y, model, fdr = 0.1, offset = 1, seed = NULL, ntrees = 1000, tune_mtry = FALSE,  mtry = NULL, metric = NULL, rftuning = FALSE, conservative = FALSE) {
  # Check for input errors
  if (!is.matrix(X) || !is.matrix(X_tilde)) stop("X and X_tilde must be matrices.")
  if (ncol(X) != ncol(X_tilde)) stop("X and X_tilde must have the same number of columns.")
  if (length(Y) != nrow(X)) stop("Length of Y must match the number of rows in X.")
  if (!(model %in% c("glmnet", "Random Forest"))) stop("Invalid model. Choose 'glmnet' or 'Random Forest'.")
  if (!is.numeric(fdr) || fdr <= 0 || fdr >= 1) stop("fdr must be a numeric value between 0 and 1.")
  if (!is.numeric(offset) || !(offset %in% c(0, 1))) stop("offset must be 0 or 1.")
  if (!is.null(seed) && (!is.numeric(seed) || seed <= 0)) stop("seed must be a positive numeric value.")
  if (model == "Random Forest" && (!is.numeric(ntrees) || ntrees <= 0)) stop("ntrees must be a positive numeric value.")
  if (model == "Random Forest" && !is.logical(tune_mtry)) stop("tune_mtry must be TRUE or FALSE.")
  if (!is.null(mtry) && (!is.numeric(mtry) || mtry <= 0)) stop("mtry must be a positive numeric value.")
  
  X_aug <- cbind(X, X_tilde)
  
  if (model == "glmnet") {
    if (is.factor(Y) || length(unique(Y)) == 2) { # Binary response
      W <- stat.lasso_coefdiff_bin(X, X_tilde, Y)
    } else { # Continuous response
      W <- stat.lasso_coefdiff(X, X_tilde, Y)
    }
    if (conservative == FALSE){
      T <- knockoff.threshold(W, fdr = fdr, offset = offset)
    } else {
      T = (1-fdr)*ko.sel(W, print = FALSE, method = "gaps")$threshold
    }
    selected_glm <- sort(which(W >= T))
    out = list(selected = selected_glm,
               W = W,
               T = T)
    return(out)
  } else if (model == "Random Forest") {
    if (rftuning == TRUE){
      if (is.null(mtry)) { # Tune mtry if not provided
        if (tune_mtry) {
          if (is.factor(Y) || length(unique(Y)) == 2) { # Binary response
            set.seed(seed)
            bestmtry <- tuneRF(X_aug, as.factor(Y), stepFactor = 1.5, improve = 1e-5, ntree = ntrees, trace = FALSE)
            mtry <- bestmtry[as.numeric(which.min(bestmtry[, "OOBError"])), 1]
          } else{
            set.seed(seed)
            bestmtry <- tuneRF(X_aug,Y, stepFactor = 1.5, improve = 1e-5, ntree = ntrees, trace = FALSE)
            mtry <- bestmtry[as.numeric(which.min(bestmtry[, "OOBError"])), 1]
          }}
        else {
          mtry <- floor(sqrt(ncol(X_aug))) # Default mtry
        }
      }
      if (is.factor(Y) || length(unique(Y)) == 2) { # Binary response
        set.seed(seed)
        model_rf <- randomForest(X_aug, as.factor(Y), ntree = ntrees, mtry = mtry, importance = TRUE)
        if (metric == "Accuracy"){
          cf <- importance(model_rf)[, 1]
        } else if (metric == "Gini"){
          cf <- importance(model_rf)[,3]
        }
      } else {
        set.seed(seed)
        model_rf <- randomForest(X_aug, Y, ntree = ntrees, mtry = mtry, importance = TRUE)
        if (metric == "Accuracy"){
          cf <- importance(model_rf)[, 1]
        } else if (metric == "Gini"){
          cf <- importance(model_rf)[,3]
        }
      }
      W <- abs(cf[1:ncol(X)]) - abs(cf[(ncol(X) + 1):ncol(X_aug)])
      if (conservative == FALSE){
        T <- knockoff.threshold(W, fdr = fdr, offset = offset)
      } else {
        T = (1-fdr)*ko.sel(W, print = FALSE, method = "gaps")$threshold
      }} else if (rftuning == FALSE){
        if (is.factor(Y) || length(unique(Y)) == 2) { # Binary response
          set.seed(seed)
          W <- stat.random_forest(X,X_tilde,as.factor(Y))
          if (conservative == FALSE){
            T <- knockoff.threshold(W, fdr = fdr, offset = offset)
          } else {
            T = (1-fdr)*ko.sel(W, print = FALSE, method = "gaps")$threshold
          }
        } else{
          set.seed(seed)
          W <- stat.random_forest(X,X_tilde,Y)
          if (conservative == FALSE){
            T <- knockoff.threshold(W, fdr = fdr, offset = offset)
          } else {
            T = (1-fdr)*ko.sel(W, print = FALSE, method = "gaps")$threshold
          }
        }
      }
    selected_rf <- sort(which(W >= T))
    out = list(selected = selected_rf,
               W = W,
               T = T)
    return(out)
  }
}

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

run_lda_iteration <- function(sim_params){
  p <- sim_params$p
  pos <- sim_params$pos
  u <- sim_params$u
  sim <- sim_params$sim
  target_fdr <- sim_params$target_fdr
  
  tryCatch({
    # Generate data
    data_sim <- generate_data_AA(p = p, pos = pos, u = u, seed = sim)
    X_sim <- data_sim$X
    Y_sim <- data_sim$Y
    true_indices <- data_sim$signal_indices    
    
    df.LDA <- as(as.matrix(X_sim), "dgCMatrix")
    vanilla.LDA <- LDA(df.LDA, k = 8, method = "VEM")   
    theta.LDA <- vanilla.LDA@gamma
    beta.LDA <- vanilla.LDA@beta
    beta.LDA <- t(apply(beta.LDA, 1, function(row) row/sum(row)))
    X_sim_tilde.LDA <- generateKnockoff(X_sim, theta.LDA, beta.LDA, seed = sim)          
    
    # LDA-KF method
    ldakf_result <- suppressWarnings(
      zinck.filter(X_sim, X_sim_tilde.LDA, as.factor(Y_sim),
                   model = "Random Forest", 
                   fdr = target_fdr,
                   offset = 1,
                   seed = sim,
                   rftuning = TRUE,
                   metric = "Gini",
                   mtry = 67)
    )
    index_est_ldakf <- ldakf_result$selected          
    
    # Calculate fdr and power for LDA-KF
    FN_ldakf <- sum(!(true_indices %in% index_est_ldakf))
    FP_ldakf <- sum(!(index_est_ldakf %in% true_indices))
    TP_ldakf <- sum(index_est_ldakf %in% true_indices)
    ldakf_fdr <- ifelse(length(index_est_ldakf) > 0,
                        FP_ldakf / (FP_ldakf+TP_ldakf), 0)
    ldakf_power <- TP_ldakf / (TP_ldakf+FN_ldakf)     
    
    return(data.frame(
      p = p,
      pos = pos,
      u = u,
      sim = sim,
      target_fdr = target_fdr,
      method = "LDA-KF",
      empirical_fdr = ldakf_fdr,
      power = ldakf_power,
      n_selected = length(index_est_ldakf)
    ))
    
  }, error = function(e){
    failed_result <- data.frame(
      p = p,
      pos = pos,
      u = u,
      sim = sim,
      target_fdr = target_fdr,
      method = "LDA-KF",
      empirical_fdr = NA,
      power = NA,
      n_selected = 0
    )
    
    warning(sprintf("Iteration failed for sim %d: %s", sim, e$message))
    return(failed_result)
  })
}
  
  
cat("begin parallel computations...\n")
n_cores <- 4  
n_iter <- 100 
start_time <- Sys.time()
  
sim_param_list <- lapply(1:n_iter, function(sim_id) {
  list(p = p, pos = pos, u = u, sim = sim_id, target_fdr = target_fdr)
  })
  
cl <- makeCluster(n_cores)
registerDoParallel(cl)
  
clusterExport(cl, c("generate_data_AA", "generateKnockoff",
                    "zinck.filter", "log_normalize", "run_lda_iteration",
                    "count"))
clusterEvalQ(cl, {
    library(topicmodels)
    library(randomForest)
    library(knockoff)
    library(Matrix)
    library(stats)
    library(kosel)
    library(lubridate)
  })
  
batch_results <- foreach(params = sim_param_list,
                         .combine = rbind,
                         .packages = c("topicmodels", "randomForest", "knockoff", 
                                       "Matrix", "stats", "kosel","lubridate"),
                         .errorhandling = "pass") %dopar% {
                           run_lda_iteration(params)
                           }
  
stopCluster(cl)
registerDoSEQ()
  
end_time <- Sys.time()
  
# save results
output_file <- sprintf("lda_results_p%d_pos%d_u%.1f_fdr%.2f.RData", p, pos, u, target_fdr)
save(batch_results, file = output_file)
  
elapsed <- as.numeric(end_time - start_time)
cat(sprintf("total complete time: %.1f h\n", elapsed/3600))
  
cat("Complete simulation for LDA-KF ends!\n")
