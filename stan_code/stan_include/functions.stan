functions {
  real logistic_hinge(real x, real x0, real a, real b0, real b1, real delta) { 
    real xdiff = x - x0;
    return a + b0 * xdiff + (b1 - b0) * delta * log1p_exp(xdiff / delta);
  }

  matrix to_fac_load(vector lambda, vector lambda_diag, int NP, int D) {
    matrix[NP, D] A;

    int idx = 0;
    for(i in 1:(D-1)) {
      for(j in (i+1):D) {
        A[i,j] = 0; 
      }
    } //0 on upper diagonal
    for(j in 1:D) {
      for(i in (j+1):NP) {
        idx += 1;
        A[i,j] = lambda[idx];
      }
      A[j,j] = lambda_diag[j];
    }
    return A;

  }

  matrix to_fac_load_lower_tri(vector lambda, vector lambda_diag, int NP, int D) {
    matrix[NP, D] A;

    int idx = 0;
    for(i in 1:(D-1)) {
      for(j in 1:i) {
        A[i+NP-D+1,j] = 0; 
      }
    } //0 on lower diagonal
    for(j in 1:D) {
      for(i in 1:(NP-D+j-1)) {
        idx += 1;
        A[i,j] = lambda[idx];
      }
      A[j+NP-D,j] = lambda_diag[j];
    }
    return A;

  }

  matrix cov2cor(matrix V) {
    int p = rows(V);
    vector[p] Is = inv_sqrt(diagonal(V));
    return quad_form_diag(V, Is);
  } 
}
