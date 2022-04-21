//static int *cps;
//static double *f;
static int *r;
static double *_costs;
static int *tmpt;


void PELT(cost_func, sumstat,n,pen,cptsout,error, shape, min_len, f, cps, n_cps)
  char **cost_func;
  double *sumstat;    /* Summary statistic for the time series */
	int *n;			/* Length of the time series */
  double *pen;  /* Penalty used to decide if a changepoint is significant */
  int *cptsout;    /* Vector of identified changepoint locations */
  int *error;   /* 0 by default, nonzero indicates error in code */
  double *shape; // only used when cost_func is the gamma likelihood 
  int *min_len; //minimum segment length 
  double *f; // stores likelihood up to that time using optimal changepoint locations up to that time 
  int *cps; // stores last changepoint locations 
  int *n_cps; //stores the current number of changepoints 
  {
	// R code does know.mean and fills mu if necessary
  
  int *r;
  r = (int *)calloc(*n+1,sizeof(int));
  
  int r_len;
  double minout;

  double *_costs;
  _costs = (double *)calloc(*n+1,sizeof(double));

  int *tmpt;
  tmpt = (int *)calloc(*n+1,sizeof(int));

  
  int tau_star,i,whichout,nchecktmp;
  
  f[0]= -*pen;
  cps[0]=0; 
  n_cps[0]=0;
  
  int j; 
  
  for(j=*min_len;j<(2*(*min_len));j++){
    f[j] = costfunction(*(sumstat+j),*(sumstat + *n + 1 + j),*(sumstat + *n + *n + 2 + j),j, *shape); 
  }
  
  
  for(j=*min_len;j<(2*(*min_len));j++){ 
    cps[j] = 0;
  }
  
   for(j=*min_len;j<(2*(*min_len));j++){ 
    n_cps[j] =1;
  }
  
  
  r_len=2;
  r[0]=0;
  r[1]=*min_len;
  
  for(tau_star=2*(*min_len);tau_star<(*n+1);tau_star++){
      if ((f[tau_star]) == 0){ 
          for(i=0;i<(r_len);i++){
              _costs[i]=f[r[i]] + costfunction(*(sumstat+tau_star)-*(sumstat+r[i]),*(sumstat + *n + 1 +tau_star)-*(sumstat + *n + 1 +r[i]),*(sumstat + *n + *n + 2 +tau_star)-*(sumstat + *n + *n + 2 +r[i]), tau_star-r[i], *shape)+*pen;
              }
            min_which(_costs,r_len,&minout,&whichout); /*updates minout and whichout with min and which element */
            f[tau_star]=minout;
            cps[tau_star]=r[whichout]; 
            n_cps[tau_star]=n_cps[cps[tau_star]]+1;

            /* Update r for next iteration, first element is next tau */
            nchecktmp=0;
            for(i=0;i<r_len;i++){
                if(_costs[i]<= (f[tau_star]+*pen)){
                    *(r+nchecktmp)=r[i];
                    nchecktmp+=1;
                    }
                }
            r_len = nchecktmp;
        }
    *(r+r_len)=tau_star-(*min_len-1);// atleast 1 obs per seg
     r_len+=1;
  
  } // end taustar
  
  // put final set of changepoints together
  int ncpts=0;
  int last=*n;
  while(last!=0){
      *(cptsout + ncpts) = last; 
      last=cps[last];
      ncpts+=1;
  }
}

