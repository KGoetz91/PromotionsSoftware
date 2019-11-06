/* formfactors*/

#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>

double sinc(double x){return sin(x)/x;}

double lognorm(double x, double N, double sigma, double mu){
  double factor = N/(sqrt(2*M_PI)*x)
  double exponent = exp(-(ln(pow((x/mu),2)))/(2*pow(sigma,2)))
  return factor*exponent
}

double ff_cube(int n, double *x, double *params){

  double eta_core, eta_solv, edge, sigma, q;
  sigma = params[0];
  eta_core = params[1];
  eta_solv = params[2];
  edge = params[3];
  q = params[4];
  
  double delta_eta = eta_core-eta_solv;
  double volume = pow(edge, 3);
  
  double theta = x[0];
  double phi = x[1];
  double res = 0;
  
  if (q < sigma){
    res = delta_eta*volume;
  }else{
    double r = edge/2;
    double qx = -q*sin(theta)*cos(phi);
    double qy = q*sin(theta)*sin(phi);
    double qz = q*cos(theta);
    
    res = delta_eta*volume*sinc(qx*r)*sinc(qy*r)*sinc(qz*r);
    
  }
  
  return res;

}

double int_cube(int n, double *x, void *user_data){

  double params[5];
  params[0] = (*(double *) user_data);
  params[1] = (*((double *) user_data +1));
  params[2] = (*((double *) user_data +2));
  params[3] = (*((double *) user_data +3));
  params[4] = (*((double *) user_data +4));
  
  double res = ff_cube(n, x, params);
  return pow(res,2);

}

double int_cube_one_shell(int n, double *x, void *user_data){
  double params1[5], params2[5];
  
  params1[0] = (*(double *) user_data);
  params1[1] = (*((double *) user_data +1));
  params1[2] = (*((double *) user_data +2));
  params1[3] = (*((double *) user_data +4));
  params1[4] = (*((double *) user_data +6));
  
  params2[0] = (*(double *) user_data);
  params2[1] = (*((double *) user_data +2));
  params2[2] = (*((double *) user_data +3));
  params2[3] = params1[3] + (*((double *) user_data +5));
  params2[4] = (*((double *) user_data +6));
  
  double res = ff_cube(n, x, params1)+ff_cube(n, x, params2);
  
  return pow(res,2);
}
