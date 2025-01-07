#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <sstream>
#include "TF1.h"
#include "TMath.h"

// par[9]  is min value
// par[10] is max value
// par[11] tells whether to reject signal region (par[11]==1), reject bkg region (par[11]==-1), or reject no region (par[11]==0 or really anything other than +/-1)
Double_t jzTwoGausSig_ExpBkg(Double_t *x, Double_t *par)
{
    // reject anything in signal region
    if (par[13]==1. && par[11] < x[0] && x[0] < par[12]) {
      TF1::RejectPoint();
      // return 0;
   }
   
   // reject anything outside signal region
    if (par[13]==-1. && ( x[0]<par[11] || x[0] > par[12]) ) {
      TF1::RejectPoint();
      // return 0;
   }
   
   Double_t sig = par[0]*((1-par[3])*TMath::Gaus(x[0],par[1],par[2],1)+par[3]*TMath::Gaus(x[0],par[4],par[5],1));
   Double_t xm  = x[0]-par[7];
   Double_t bkg = par[6]*TMath::Exp(par[8]*xm+par[9]*xm*xm+par[10]*xm*xm*xm );
   
   return sig+bkg;
}


