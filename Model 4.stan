data {
  // 1) Basic 
  int<lower=0> N;                 //  number of training observations
  int<lower=0> N_test;            //  number of test observations
  int<lower=1> K;                 //  number of columns in X
  int<lower=2> J_mode;            //  number of travel modes
  int<lower=2> J_country;         //  number of countries

  matrix[N, K] X_train;           // training design matrix
  matrix[N_test, K] X_test;       // test design matrix

  array[N] int<lower=1, upper=J_mode>    Y_train_mode;    
  array[N] int<lower=1, upper=J_country> Y_train_country; 
  array[N_test] int<lower=1, upper=J_mode>    Y_test_mode;    
  array[N_test] int<lower=1, upper=J_country> Y_test_country; 

  // 2) Year index
  int<lower=1> N_years;           // number of training years
  int<lower=1> N_years_total;     // number of total years in both training and test
  array[N] int<lower=1, upper=N_years>       Year_train_idx;
  array[N_test] int<lower=1, upper=N_years_total> Year_test_idx;

  // 3) Log-HHINK imputation data
  int<lower=0> N_all;        // total obs
  int<lower=0> N_obs;        // with observed log(HHINK)
  int<lower=0> N_miss;       // with missing log(HHINK)
  array[N_obs] int HHINK_obs_idx;    // row indices of observed
  array[N_miss] int HHINK_miss_idx;  // row indices of missing
  vector[N_obs] logHHINK_obs;        // observed log(HHINK) values
}

transformed data {
  real logHHINK_mean;
  real logHHINK_sd;
  vector[N_obs] logHHINK_obs_scaled;

  logHHINK_mean = mean(logHHINK_obs);
  logHHINK_sd = sd(logHHINK_obs);

  for (i in 1:N_obs)
    logHHINK_obs_scaled[i] = (logHHINK_obs[i] - logHHINK_mean) / logHHINK_sd;
}

parameters {
  // 1) mode choice
  matrix[K, J_mode - 1] beta_mode;           
  array[J_mode] simplex[J_country] theta_mode;

  // 2) Year intercept random walk (non-centered)
  real<lower=0> tau;                     // step size
  row_vector[J_mode - 1] alpha_init;     // intercepts for year 1 (row_vector)
  matrix[N_years - 1, J_mode - 1] phi;   // standard normal innovations (rows also row_vectors)

  // 3) Log-HHINK imputation parameter 
  vector[N_miss] logHHINK_impute_scaled;   

  // 4) Coefficients for standardized log(HHINK)
  vector[J_mode - 1] beta_hhink;  
  
  // 5) Dispersion parameter per mode for country choice
  array[J_mode] real <lower=0> kappa;
}

transformed parameters {
  // Full year intercept matrix
  matrix[N_years, J_mode - 1] alpha_mode;

  // random walk
  alpha_mode[1, ] = alpha_init;
  for (t in 2:N_years) {
    alpha_mode[t, ] = alpha_mode[t - 1, ] + tau * phi[t - 1, ];
  }

  // Full effect log(HHINK)
  vector[N_all] logHHINK_scaled;     

  // Merge observed + imputed
  logHHINK_scaled[HHINK_obs_idx] = logHHINK_obs_scaled;
  logHHINK_scaled[HHINK_miss_idx] = logHHINK_impute_scaled;

}

model {
  // (1) Priors on coefficients
  to_vector(beta_mode) ~ normal(0, 2);
  beta_hhink           ~ normal(0, 2);

  // (2) Dirichlet prior for mode to country
  for (m in 1:J_mode){
    kappa[m] ~ inv_gamma(2,1);
    theta_mode[m] ~ dirichlet(rep_vector(kappa[m], J_country));
  }

  // (3) Non-centered random walk
  tau            ~ normal(0, 1);
  alpha_init     ~ normal(0, 1);
  to_vector(phi) ~ normal(0, 1);

  // (4) Non-centered Log-HHINK imputation
  logHHINK_impute_scaled      ~ normal(0, 1);   

  // (5) Likelihood: Mode choice 
  {
    matrix[N, J_mode] eta_mode_train;
    for (n in 1:N) {
      int t = Year_train_idx[n];
      for (j in 1:(J_mode - 1)) {
        eta_mode_train[n, j] =
          dot_product(X_train[n], beta_mode[, j]) +
          beta_hhink[j]    * logHHINK_scaled[n] +
          alpha_mode[t, j];
      }
      eta_mode_train[n, J_mode] = 0; // baseline
    }
    for (n in 1:N)
      Y_train_mode[n] ~ categorical_logit(eta_mode_train[n]');
  }

  // (6) Likelihood: Country choice
  for (n in 1:N) {
    Y_train_country[n] ~ categorical(theta_mode[Y_train_mode[n]]);
  }
}

generated quantities {
  /* 0.  Extended intercept matrix  */
  matrix[N_years_total, J_mode - 1] alpha_mode_ext;   // ← add name + “;”

  /* 1.  Per-person posterior-predictive draws */
  array[N_test] int<lower=1, upper=J_mode>    mode_pred;
  array[N_test] int<lower=1, upper=J_country> country_pred;

  /* 2.  Year-aggregated counts  */
  int<lower=0> N_mode_year_pred[N_years_total, J_mode];
  int<lower=0> N_country_year_pred[N_years_total, J_country];

  /* 3.  Log-likelihoods  */
  vector[N_test] log_lik_mode;
  vector[N_test] log_lik_country;

  /* zero-initialise the new, larger arrays */
  for (t in 1:N_years_total) {
    for (m in 1:J_mode)
      N_mode_year_pred[t, m] = 0;
    for (k in 1:J_country)
      N_country_year_pred[t, k] = 0;
  }

  /* ---------- build alpha_mode_ext ---------- */
  for (t in 1:N_years)
  alpha_mode_ext[t] = alpha_mode[t];              

  for (t in (N_years + 1):N_years_total) {         
    row_vector[J_mode - 1] z;
    for (j in 1:(J_mode - 1))
      z[j] = normal_rng(0, 1);
   alpha_mode_ext[t] = alpha_mode_ext[t - 1] + tau * z;
  }

  /* ---------- predictions ---------- */
  for (n in 1:N_test) {
    int  t = Year_test_idx[n];
    vector[J_mode] eta_mode;

    for (j in 1:(J_mode - 1)) {
      eta_mode[j] =
        dot_product(X_test[n], beta_mode[, j]) +
        beta_hhink[j] * logHHINK_scaled[N + n] +
        alpha_mode_ext[t, j]; 
    }
    eta_mode[J_mode] = 0;

    mode_pred[n]    = categorical_logit_rng(eta_mode);
    country_pred[n] = categorical_rng(theta_mode[mode_pred[n]]);

    log_lik_mode[n]    = categorical_logit_lpmf(Y_test_mode[n]    | eta_mode);
    log_lik_country[n] = categorical_lpmf     (Y_test_country[n]  | theta_mode[Y_test_mode[n]]);

    N_mode_year_pred[t,    mode_pred[n]]    += 1;
    N_country_year_pred[t, country_pred[n]] += 1;
  }
}
