/*********************************************
 ********************************************/

data {
  int N;  // number of records
  int NEQ;  // number of earthquakes
  int NSTAT;  // number of stations
  int K; // number of coefficients - should be 7 or 9
  int NP; // umber of periods

  vector[N] M;
  vector[N] R;
  vector[N] VS;
  vector[N] FS;
  vector[N] FR;

  matrix[N, NP] Y; // ln ground-motion value

  array[N] int<lower=1,upper=NEQ> eq; // event id (in numerical order from 1 to last)
  array[N] int<lower=1,upper=NSTAT> stat; // station id (in numerical order from 1 to last)

  vector<lower=0>[3] alpha; // dirichlet hyperparameter

  array[NP] int<lower=1,upper=N> len; // umber of useable records
  array[N, NP] int<lower=0,upper=N> idx; // indices of useable records

  array[NP] vector[K] mu_prior;
  array[NP] vector[K] sd_prior;

  // fixed coefficients
  vector[NP] Mh;
  vector[NP] Mref;
  vector[NP] h;

}

transformed data {
  real delta = 1e-9;
  real ln05 = log(0.3);
  real sig = 0.5;

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

  vector<lower=0>[NP] sigma_total;
  array[NP] simplex[3] omega;

  matrix[NEQ, NP] eqterm;
  matrix[NSTAT, NP] statterm;
}

transformed parameters {
  vector[NP] tau_0;
  vector[NP] phi_0;
  vector[NP] phi_S2S;

  for(p in 1:NP) {
    phi_0[p] = sqrt(omega[p, 1] * square(sigma_total[p]));
    tau_0[p] = sqrt(omega[p, 2] * square(sigma_total[p]));
    phi_S2S[p] = sqrt(omega[p, 3] * square(sigma_total[p]));
  }
}

model {
  ic ~ normal(mu_prior[:,1], sd_prior[:,1]);
  for(p in 1:NP) {
    eqterm[:,p] ~ normal(0, tau_0[p]);
    statterm[:,p] ~ normal(0, phi_S2S[p]);
    omega[p] ~ dirichlet(alpha);
    c_qr[p] ~ std_normal();

    vector[N] mu = ic[p] + Q_ast[p] * c_qr[p] + eqterm[eq,p] + statterm[stat,p];
    Y[idx[1:len[p],p],p] ~ normal(mu[idx[1:len[p],p]], phi_0[p]);
  }
}

generated quantities {
  array[NP] vector[K] c;
  for(p in 1:NP) {
    c[p,1] = ic[p];
    c[p,2:K] =  R_ast_inverse[p] * c_qr[p];
  }
}
