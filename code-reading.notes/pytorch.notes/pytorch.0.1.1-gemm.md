```c++
void THBlas_(gemm)(char transa, char transb, long m, long n, long k, real alpha, real *a, long lda, real *b, long ldb, real beta, real *c, long ldc)
{
  int transa_ = ((transa == 't') || (transa == 'T'));
  int transb_ = ((transb == 't') || (transb == 'T'));

  if(n == 1)
    ldc = m;

  if(transa_)
  {
    if(m == 1)
      lda = k;
  }
  else
  {
    if(k == 1)
      lda = m;
  }

  if(transb_)
  {
    if(k == 1)
      ldb = n;
  }
  else
  {
    if(n == 1)
      ldb = k;
  }

#if defined(USE_BLAS) && (defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT))
  if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (lda <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_k = (int)k;
    int i_lda = (int)lda;
    int i_ldb = (int)ldb;
    int i_ldc = (int)ldc;

#if defined(TH_REAL_IS_DOUBLE)
    dgemm_(&transa, &transb, &i_m, &i_n, &i_k, &alpha, a, &i_lda, b, &i_ldb, &beta, c, &i_ldc);
#else
    sgemm_(&transa, &transb, &i_m, &i_n, &i_k, &alpha, a, &i_lda, b, &i_ldb, &beta, c, &i_ldc);
#endif
    return;
  }
#endif
  {
    long i, j, l;
    if(!transa_ && !transb_)
    {
      real *a_ = a;
      for(i = 0; i < m; i++)
      {
        real *b_ = b;
        for(j = 0; j < n; j++)
        {
          real sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l*lda]*b_[l];
          b_ += ldb;
	  if (beta == 0)
	    c[j*ldc+i] = alpha*sum;
	  else
	    c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_++;
      }
    }
    else if(transa_ && !transb_)
    {
      real *a_ = a;
      for(i = 0; i < m; i++)
      {
        real *b_ = b;
        for(j = 0; j < n; j++)
        {
          real sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l]*b_[l];
          b_ += ldb;
	  if (beta == 0)
	    c[j*ldc+i] = alpha*sum;
	  else
	    c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_ += lda;
      }
    }
    else if(!transa_ && transb_)
    {
      real *a_ = a;
      for(i = 0; i < m; i++)
      {
        real *b_ = b;
        for(j = 0; j < n; j++)
        {
          real sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l*lda]*b_[l*ldb];
          b_++;
	  if (beta == 0)
	    c[j*ldc+i] = alpha*sum;
	  else
	    c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_++;
      }
    }
    else
    {
      real *a_ = a;
      for(i = 0; i < m; i++)
      {
        real *b_ = b;
        for(j = 0; j < n; j++)
        {
          real sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l]*b_[l*ldb];
          b_++;
	  if (beta == 0)
	    c[j*ldc+i] = alpha*sum;
	  else
	    c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_ += lda;
      }
    }
  }
}
```
