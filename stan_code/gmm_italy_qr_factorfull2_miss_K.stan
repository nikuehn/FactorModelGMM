/*
 * marginal for record level
 * */

#include functions.stan

data {
  int N;  // number of records
  int NEQ;  // number of earthquakes
  int NSTAT;  // number of stations
  int K; // number of coefficients - should be 9
  int NP; // umber of periods
  int<lower=1,upper=NP> Pm; // period index for which all records are
  int<lower=1,upper=NP> D; // number of factors

  vector[N] M;
  vector[N] R;
  vector[N] VS;
  vector[N] FS;
  vector[N] FR;

  array[N] vector[NP] Y; // ln ground-motion value

  array[N] int<lower=1,upper=NEQ> eq; // event id (in numerical order from 1 to last)
  array[N] int<lower=1,upper=NSTAT> stat; // station id (in numerical order from 1 to last)

  array[NP] int<lower=0,upper=N> len_per; // umber of useable records
  array[N, NP] int<lower=0,upper=N> idx_per; // indices of useable records

  array[NP] vector[K] mu_prior;
  array[NP] vector[K] sd_prior;

  // fixed coefficients
  vector[NP] Mh;
  vector[NP] Mref;
  vector[NP] h;
}

transformed data {
  real delta = 1e-9;
  real ln02 = log(0.2);
  real sig = 0.5;
  int D_lt = D * (NP - D) + (D * (D - 1)) %/% 2; // number of lower-triangular, non-zero loadings

  array[NP] matrix[N, K-1] X;
  array[NP] matrix[N, K-1] Q_ast;
  array[NP] matrix[K, K-1] R_ast;
  array[NP]matrix[K-1, K-1] R_ast_inverse;

  for(p in 1:NP) {
    for(i in 1:N) {
      // magnitude scaling
      X[p, i, 1] = (M[i] <= Mh[p]) ?  (M[i] - Mh[p]) : 0;
      X[p, i, 2] = (M[i] > Mh[p]) ?  (M[i] - Mh[p]) : 0;

      // distance scaling
      X[p, i, 3] = (M[i] - Mref[p]) * log10(sqrt(square(R[i]) + square(h[p])));
      X[p, i, 4] = log10(sqrt(square(R[i]) + square(h[p])));
      X[p, i, 5] = sqrt(square(R[i]) + square(h[p]));

      if(K == 9) {
        // fault type;
        X[p, i, 6] = FS[i];
        X[p, i, 7] = FR[i];

        // Vs30
        X[p, i, 8] = (VS[i] <= 1500) ? log10(VS[i] / 800.) : log10(1500. / 800.);
      }
      else {
        // Vs30
        X[p, i, 6] = (VS[i] <= 1500) ? log10(VS[i] / 800.) : log10(1500. / 800.);
      }
    }
    Q_ast[p] = qr_thin_Q(X[p]) * sqrt(N - 1);
    R_ast[p] = qr_thin_R(X[p]) / sqrt(N - 1);
    R_ast_inverse[p] = inverse(R_ast[p]);
  }
}

parameters {
  array[NP] vector[K-1] c_qr;
  vector[NP] ic;

  vector<lower=0>[NP] phi_0;

  vector[D_lt] lambda_eq; // Lower diagonal loadings
  vector<lower=0>[D] lambda_eq_diag; // Lower diagonal loadings
  vector[D_lt] lambda_stat; // Lower diagonal loadings
  vector<lower=0>[D] lambda_stat_diag; // Lower diagonal loadings
  vector[D_lt] lambda_rec; // Lower diagonal loadings
  vector<lower=0>[D] lambda_rec_diag; // Lower diagonal loadings

  matrix[D, NEQ] eqterm_fac;
  matrix[D, NSTAT] statterm_fac;
}

transformed parameters {
  matrix[NP, D] A_eq = to_fac_load_lower_tri(lambda_eq, lambda_eq_diag, NP, D); // factor loading matrix
  matrix[NP, D] A_stat = to_fac_load_lower_tri(lambda_stat, lambda_stat_diag, NP, D); // factor loading matrix
  matrix[NP, D] A_rec = to_fac_load_lower_tri(lambda_rec, lambda_rec_diag, NP, D); // factor loading matrix

  matrix[NEQ, NP] eqterm = (A_eq * eqterm_fac)';
  matrix[NSTAT, NP] statterm = (A_stat * statterm_fac)';
}

model {
  phi_0 ~ inv_gamma(5,0.2);
  ic ~ normal(mu_prior[:,1], sd_prior[:,1]);

  lambda_eq ~ normal(0,0.5);
  lambda_eq_diag ~ normal(0,0.5);
  lambda_stat ~ normal(0,0.5);
  lambda_stat_diag ~ normal(0,0.5);
  lambda_rec ~ normal(0,0.5);
  lambda_rec_diag ~ normal(0,0.5);

  to_vector(eqterm_fac) ~ std_normal();
  to_vector(statterm_fac) ~ std_normal();

  array[N] vector[NP] mu;
  for(p in 1:NP) {
    c_qr[p] ~ std_normal();

    mu[:,p] = to_array_1d(ic[p] + Q_ast[p] * c_qr[p] + eqterm[eq,p] + statterm[stat,p]);
  }

  matrix[NP, NP] L = cholesky_decompose(add_diag(A_rec * A_rec', square(phi_0)));
  Y[idx_per[1:len_per[1],1]] ~ multi_normal_cholesky(mu[idx_per[1:len_per[1],1]], L);
  for(p in Pm:(NP - 1)) {
    Y[idx_per[1:len_per[p],p],1:p] ~ multi_normal_cholesky(mu[idx_per[1:len_per[p],p],1:p], block(L, 1, 1, p, p));

  }
}

generated quantities {
  matrix[NP,NP] C_eq = cov2cor(A_eq * A_eq');
  matrix[NP,NP] C_stat = cov2cor(A_stat * A_stat');
  matrix[NP,NP] C_rec = cov2cor(A_rec * A_rec');
  vector[NP] tau = sqrt(diagonal(A_eq * A_eq'));
  vector[NP] phi_S2S = sqrt(diagonal(A_stat * A_stat'));
  vector[NP] phi = sqrt(diagonal(A_rec * A_rec') + square(phi_0));

  array[NP] vector[K] c;
  for(p in 1:NP) {
    c[p,1] = ic[p];
    c[p,2:K] =  R_ast_inverse[p] * c_qr[p];
  }
}
