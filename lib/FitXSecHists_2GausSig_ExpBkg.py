#!/usr/bin/env python3

#PyRoot file created from template located at:

import os, sys, time
from array import array
from math import sqrt
from ROOT import TF1,TH1F,TCanvas,TDirectoryFile,TFitResultPtr,TPaveStats
from ROOT import gROOT,gSystem,gStyle,gDirectory

kDashed = 2
kRed = 632
kDarkRed = 632+2
kOrange = 802
kYellow = 400
kGreen = 416
kDarkGreen = 416+3
kBlue = 600
kViolet = 880
kMagenta = 616
kBlack = 1
kCyan = 432
kGray    = 920
kPink   = 900
kAzure   =  860
kTeal   = 840
kWhite  = 0


class jzHistFitter:

	NUM_PARMS_SIG = 6
	NUM_PARMS_BKG = 5
	pi=3.14159

	def __init__(self,CXX_DIR,rebin_factor,
	             fitRangeLo,fitRangeHi,sigRejectRangeLo,sigRejectRangeHi,sigIntegrateRangeLo,sigIntegrateRangeHi,
				 canvas_tag,xaxis_title=""):
		# From arguments read in
		self.rebin_factor   = rebin_factor
		self.canvas_tag    = canvas_tag
		self.xaxis_title    = xaxis_title
		# STUFF DEFINED HERE!
		self.CXX_DIR        = CXX_DIR
		self.FITTED_HIST_PLOTDIR        = ""
		self.VERBOSE        = True
		self.UseLikFit      = True
		self.SaveCanvasLvl  = 0 # 0=none saved, 1=save default hist, 2=save all fit variations
		self.fitTimer = 0. # Time spent just on the fitting 
		self.startTime = 0. # Track time spent from start to finish
		self.endTime = 0.   # Track time spent from start to finish
		self.max_nfits     = 2 # Refit if total/signal uncertainty too large compared to sqrt(N) or nonzero return code
		self.sqrtN_tolerance,self.sqrtN_sig_tolerance = 3.,5. # times sqrt(N)
		self.max_chi2 = 20. if not "{+}" in xaxis_title else 100. # return None if larger than this instead of dict
		# For importing my compiled function
		# Fit ranges (unpacked tuple is best)
		self.fitRangeLo=fitRangeLo
		self.fitRangeHi=fitRangeHi
		self.sigRejectRangeLo=sigRejectRangeLo
		self.sigRejectRangeHi=sigRejectRangeHi
		self.sigIntegrateRangeLo=sigIntegrateRangeLo
		self.sigIntegrateRangeHi=sigIntegrateRangeHi
		self.min_integral_tofit = 3. if not "{+}" in xaxis_title else 50. # If fewer entries than this in range, skip!

	def SetSaveHists(self,lvl,outdir):
		self.SaveCanvasLvl = lvl
		self.FITTED_HIST_PLOTDIR = outdir

	def FitHist3Cases(self,hist_orig,parm_list,variation_str,isDATAHist=True):
		
		if(self.VERBOSE): print("Fitting for variation: " + variation_str)
		
		# initial setup
		h = hist_orig.Clone() #Clone, because otherwise we'd be altering the histogram and change binning
		h.Rebin(self.rebin_factor)
		bin_width = h.GetBinWidth(0)
		old_integral = h.Integral()
		self.SetNegativeBinsToZero(h)
		new_integral = h.Integral()
		# if(self.VERBOSE): print "Setting negative bins to zero for hist: " + str(h.GetName()[2:])
		# if(self.VERBOSE): print "old_integral: " + str(old_integral) + " new integral " + str(new_integral)
		hasEnoughEntries = True
		if(self.jzHistogramSumIntegral(h,self.sigIntegrateRangeLo,self.sigIntegrateRangeHi) < self.min_integral_tofit):
			if(self.VERBOSE): print("Skipping fitting for empty/near-empty hist: " + h.GetName())
			hasEnoughEntries = False
	
		# return a dict which should match our dataframe columns
		return_dict ={"variation":[variation_str]}
	
		#Check that inputs make sense
		for parmSet in parm_list:
			if(len(parmSet)!=5):
				print("ERROR: parameter list does not contain correct number of elements! Length expected: 5.  List: " + str(parm_list))
				sys.exit()
		if(len(parm_list)!=self.NUM_PARMS_SIG+self.NUM_PARMS_BKG): 
			print("ERROR: parameter list does not have expected number of params! Num expected: " + str(self.NUM_PARMS_SIG+self.NUM_PARMS_BKG) + " Num found in list: " + str(len(parm_list)))
			sys.exit()

		# CASE 0: fit only in the background region
		my_tot_fit = self.jz_FitFunctionSetup(None,parm_list,0) # None:passed instead of existing tf1. Set initial paramters, ranges, names, etc
		bkg_yield=0.
		if(isDATAHist):
			fit_result,dummy = self.jz_PerformFitEtaHist(h,my_tot_fit,parm_list,"FitBkgRegion",variation_str)
			# Save relevant info
			if((fit_result==0 or fit_result==4) and my_tot_fit.GetNDF()>0.001):
				ErrSum                      = self.jzHistogramErrSumIntegral(h,self.sigIntegrateRangeLo,self.sigIntegrateRangeHi)
				bkg_yield                   = my_tot_fit.Integral(self.sigIntegrateRangeLo,self.sigIntegrateRangeHi)/bin_width
				bkg_yield_err               = my_tot_fit.IntegralError(self.sigIntegrateRangeLo,self.sigIntegrateRangeHi)/bin_width
				counts_above_bkgfit         = self.jzHistogramSumIntegral(h,self.sigIntegrateRangeLo,self.sigIntegrateRangeHi)-bkg_yield
				counts_above_bkgfit_err     = 0.
				try: counts_above_bkgfit_err     = sqrt(abs(ErrSum**2 - bkg_yield_err**2))
				except OverflowError:
					print("Warning: OverflowError encountered in determining Case0 fit uncertainty, setting to 0...")
					print("(note that by default bin-sum, not fit uncertainty is used)")
				return_dict["Case0_SigYield"],  return_dict["Case0_SigFitErr"],return_dict["BkgYield_Case0"],return_dict["Case0_HistSumErrSigBkg"],return_dict["Case0_Chi2NDF"] = [counts_above_bkgfit],[counts_above_bkgfit_err],[bkg_yield],[ErrSum],[my_tot_fit.GetChisquare()/my_tot_fit.GetNDF()]
			else: return_dict["Case0_SigYield"],return_dict["Case0_SigFitErr"],return_dict["BkgYield_Case0"],return_dict["Case0_HistSumErrSigBkg"],return_dict["Case0_Chi2NDF"] = [0.],[0.],[0.],[0.],[0.]
		if(not isDATAHist or not hasEnoughEntries): return_dict["Case0_SigYield"],return_dict["Case0_SigFitErr"],return_dict["BkgYield_Case0"],return_dict["Case0_HistSumErrSigBkg"],return_dict["Case0_Chi2NDF"] = [0.],[0.],[0.],[0.],[0.]
		
		# CASE 1: fit only in the background region
		my_tot_fit = self.jz_FitFunctionSetup(my_tot_fit,parm_list[0:self.NUM_PARMS_SIG],NumParmsToSet=len(parm_list[0:self.NUM_PARMS_SIG])) #Reset signal parms (use signal portions of parm_list)
		fit_result,chi2Uncert_case1 = self.jz_PerformFitEtaHist(h,my_tot_fit,parm_list[0:self.NUM_PARMS_SIG],"FitSigRegion",variation_str)
		ErrSum                      = self.jzHistogramErrSumIntegral(h,self.sigIntegrateRangeLo,self.sigIntegrateRangeHi)
		# Save relevant info
		if((fit_result==0 or fit_result==4) and my_tot_fit.GetNDF()>0.001 and hasEnoughEntries): return_dict["Case1_SigYield"],return_dict["Case1_SigFitErr"],return_dict["BkgYield_Case1"],return_dict["Case1_HistSumErrSigBkg"],return_dict["Case1_Chi2NDF"] = [my_tot_fit.GetParameter(0)/bin_width],[chi2Uncert_case1],[bkg_yield],[ErrSum],[my_tot_fit.GetChisquare()/my_tot_fit.GetNDF()]
		else:                                                                                    return_dict["Case1_SigYield"],return_dict["Case1_SigFitErr"],return_dict["BkgYield_Case1"],return_dict["Case1_HistSumErrSigBkg"],return_dict["Case1_Chi2NDF"] = [0.],[0.],[0.],[0.],[0.]
		if(not isDATAHist):
			return_dict["MC_gausMean1"]  = my_tot_fit.GetParameter(1)
			return_dict["MC_Sigma1"]     = my_tot_fit.GetParameter(2)
			return_dict["MC_gausMean2"]  = my_tot_fit.GetParameter(4)
			return_dict["MC_Sigma2"]     = my_tot_fit.GetParameter(5)
			return_dict["MC_gausFrac"]   = my_tot_fit.GetParameter(3)
		
		# CASE 2: fit only in the background region
		# No self.jz_FitFunctionSetup needed this time
		if(isDATAHist):
			# print "PARMS FOR CASE2: "
			# for parm in parm_list: print str(parm)
			fit_result,chi2Uncert_case2,bkg_yield = self.jz_PerformFitEtaHist(h,my_tot_fit,parm_list,"FitFullRegion",variation_str)
			ErrSum                                = self.jzHistogramErrSumIntegral(h,self.sigIntegrateRangeLo,self.sigIntegrateRangeHi)
			GausSigmaLarger                       = max(my_tot_fit.GetParameter(2),my_tot_fit.GetParameter(5))
			# print "GausSigma1: " + str(my_tot_fit.GetParameter(2)) + " GausSigma2 " + str(my_tot_fit.GetParameter(5))
			# Save relevant info
			if((fit_result==0 or fit_result==4) and my_tot_fit.GetNDF()>0.001 and hasEnoughEntries): return_dict["Case2_SigYield"],return_dict["Case2_SigFitErr"],return_dict["BkgYield_Case2"],return_dict["Case2_HistSumErrSigBkg"],return_dict["Case2_Chi2NDF"],return_dict["Case2_GausSigmaLarger"] = [my_tot_fit.GetParameter(0)/bin_width],[chi2Uncert_case2],[bkg_yield],[ErrSum],[my_tot_fit.GetChisquare()/my_tot_fit.GetNDF()],[GausSigmaLarger]
			else:                                                                                    return_dict["Case2_SigYield"],return_dict["Case2_SigFitErr"],return_dict["BkgYield_Case2"],return_dict["Case2_HistSumErrSigBkg"],return_dict["Case2_Chi2NDF"],return_dict["Case2_GausSigmaLarger"] = [0.],[0.],[0.],[0.],[0.],[0.]
		else:                                                                                        return_dict["Case2_SigYield"],return_dict["Case2_SigFitErr"],return_dict["BkgYield_Case2"],return_dict["Case2_HistSumErrSigBkg"],return_dict["Case2_Chi2NDF"],return_dict["Case2_GausSigmaLarger"] = [0.],[0.],[0.],[0.],[0.],[0.]
		
		# Check that we're not too close to any fit function parameter limits
		smallest_mean=min(my_tot_fit.GetParameter(1),my_tot_fit.GetParameter(4))
		largest_mean=max(my_tot_fit.GetParameter(1),my_tot_fit.GetParameter(4))
		smallest_sigma=min(my_tot_fit.GetParameter(2),my_tot_fit.GetParameter(5))
		largest_sigma=max(my_tot_fit.GetParameter(2),my_tot_fit.GetParameter(5))
		# # Check that signal parameters not too close to fit limits
		if(smallest_mean-min(parm_list[1][2],parm_list[1+3][2])<0.002 and not parm_list[1][4]): # Check mean min
			if(self.VERBOSE): print("WARNING: parameter very close to min allowed value! GAUS_MEAN-PARM_MIN="+str(smallest_mean-parm_list[1][2])) 
		if(max(parm_list[1][3],parm_list[1+3][3])-largest_mean<0.002 and not parm_list[1][4]): # Check mean max
			if(self.VERBOSE): print("WARNING: parameter very close to max allowed value! PARM_MAX-GAUS_MEAN="+str(parm_list[1][3]-largest_mean)) 
		if((smallest_sigma-min(parm_list[2][2],parm_list[2+3][2]))<0.002 and not parm_list[2][4]): # Check sigma min
			if(self.VERBOSE): print("WARNING: parameter very close to min allowed value! GAUS_SIGMA-PARM_MIN="+str(smallest_sigma-parm_list[2][2])) 
		if(max(parm_list[2][3],parm_list[2+3][3])-largest_sigma<0.002 and not parm_list[2][4]): # Check sigma max
			if(self.VERBOSE): print("WARNING: parameter very close to max allowed value! PARM_MAX-GAUS_SIGMA="+str(parm_list[2][3]-largest_sigma)) 

		del h
		del my_tot_fit

		# Skip this fit if any case larger than our chi2 threshold
		if return_dict["Case0_Chi2NDF"][0] > self.max_chi2 or return_dict["Case1_Chi2NDF"][0] > self.max_chi2 or return_dict["Case2_Chi2NDF"][0] > self.max_chi2:
			print("WARNING: FIT CHI2 FAILED FOR HIST "+hist_orig.GetName() + " and variation " + variation_str)
			print("tolerance: " + str(self.max_chi2))
			print("case 0: " + str(return_dict["Case0_Chi2NDF"]))
			print("case 1: " + str(return_dict["Case1_Chi2NDF"]))
			print("case 2: " + str(return_dict["Case2_Chi2NDF"]))
			return None

		# return a dict which should match our dataframe columns
		return return_dict
		
		
	def jz_PerformFitEtaHist(self,h,tf1,parm_list,case_str,var_tag):

		# Fix/release parameters depending on fitting case
		if(case_str=="FitBkgRegion"):
			tf1.FixParameter(self.NUM_PARMS_SIG+self.NUM_PARMS_BKG+2,1) # Setting this to 1 ==> reject signal region
			#Constrain signal to 0.
			for i in range(0,self.NUM_PARMS_SIG): tf1.FixParameter(i,tf1.GetParameter(i)) #Fix signal parameters 
			tf1.SetParameter(0,0.) # Set signal amplitude to 0
			tf1.SetParameter(3,0.) # Set signal amplitude to 0
		elif(case_str=="FitSigRegion"):
			for i in range(self.NUM_PARMS_SIG,self.NUM_PARMS_SIG+self.NUM_PARMS_BKG): tf1.FixParameter(i,tf1.GetParameter(i)) #Fix bkg parameters 
			tf1.FixParameter(self.NUM_PARMS_SIG+self.NUM_PARMS_BKG+2,-1.) # Setting this to -1 ==> fit ONLY signal region
		elif(case_str=="FitFullRegion"):
			for i in range(self.NUM_PARMS_SIG,self.NUM_PARMS_SIG+self.NUM_PARMS_BKG): 
				if(not parm_list[i][4]): tf1.ReleaseParameter(i) # Release all bkg parameters, unless we specified they should be fixed
			tf1.FixParameter(self.NUM_PARMS_SIG+self.NUM_PARMS_BKG+2,0.) # Setting this to 0 ==> fit whole region of signal and bkg
		else:
			print("ERROR: not sure which case to fit for")
			print("case_str provided: " + case_str)
			sys.exit()
		
		# Perform fit (without drawing results yet)
		# # Errors with WL fitting are a bit unstable, repeat fitting step until acceptable (or max fits reached)
		# Fit options to use or at least consider: "E" for Minos errors, "M" improve fit results, "S" to get fit results, "0" to not plot fit results (yet)
		fit_start_time = time.clock()
		fit_result = TFitResultPtr()
		fit_result = h.Fit(tf1,"QS0","",self.fitRangeLo,self.fitRangeHi) #Option "S": save fit results to TFitResultPtr. Needed for calculating yield integral / uncertainty covariance matrix	
		chi2Uncert = tf1.GetParError(0)/h.GetBinWidth(0)
		if(self.UseLikFit):
			for n in range(self.max_nfits):
				# print "WL fit number " + str(n)
				# print "Fit result: " + str(int(fit_result))
				# print "FITTING FOR CASE: " + case_str
				with suppress_stdout_stderr():
					fit_result = h.Fit(tf1,"SWL0","",self.fitRangeLo,self.fitRangeHi)
					totyield_over_sqrtn = 0
					if(tf1.Integral(self.fitRangeLo,self.fitRangeHi)/h.GetBinWidth(0) > 0. ):  totyield_over_sqrtn = (tf1.IntegralError(self.fitRangeLo,self.fitRangeHi)/h.GetBinWidth(0)) / sqrt(tf1.Integral(self.fitRangeLo,self.fitRangeHi)/h.GetBinWidth(0))
					sigyield_over_sqrtn = 0
					if(not case_str=="FitBkgRegion" and tf1.GetParameter(0)>0.0001): sigyield_over_sqrtn = (tf1.GetParError(0)/h.GetBinWidth(0) / sqrt(tf1.GetParameter(0)/h.GetBinWidth(0)))
					# If the error compared to sqrt(N) seems "good enough" then stop, otherwise we keep refitting. Doesn't seem to get worse, so taking last fit is ok.
					if(totyield_over_sqrtn<self.sqrtN_tolerance and sigyield_over_sqrtn<self.sqrtN_sig_tolerance and int(fit_result)==0): break
		# If chi2 fitting, and fit result is nonzero
		elif(fit_result!=0): 
			for n in range(self.max_nfits):
				with suppress_stdout_stderr():
					fit_result = h.Fit(tf1,"QS0","",self.fitRangeLo,self.fitRangeHi)
					totyield_over_sqrtn = 0.
					if(tf1.Integral(self.fitRangeLo,self.fitRangeHi)/h.GetBinWidth(0) > 0. ): totyield_over_sqrtn = (tf1.IntegralError(self.fitRangeLo,self.fitRangeHi)/h.GetBinWidth(0)) / sqrt(tf1.Integral(self.fitRangeLo,self.fitRangeHi)/h.GetBinWidth(0))
					sigyield_over_sqrtn = 0
					if(not case_str=="FitBkgRegion" and tf1.GetParameter(0)>0.0001): sigyield_over_sqrtn = (tf1.GetParError(0)/h.GetBinWidth(0) / sqrt(tf1.GetParameter(0)/h.GetBinWidth(0)))
					# If the error compared to sqrt(N) seems "good enough" then stop, otherwise we keep refitting. Doesn't seem to get worse, so taking last fit is ok.
					if(totyield_over_sqrtn<self.sqrtN_tolerance and sigyield_over_sqrtn<self.sqrtN_sig_tolerance and int(fit_result)==0): break
		# Check fit result status & fit duration
		status = int(fit_result)
		if(status==4): 
			print("WARNING fit status 4: " + str(status))
			print("Histogram name: " + h.GetName() + " tagname: " + str(var_tag))
		if(status!=0 and status!=4): 
			print("ERROR fit status nonzero: " + str(status))
			print("Histogram name: " + h.GetName() + " tagname: " + str(var_tag))
		fit_end_time = time.clock()
		self.fitTimer+=fit_end_time-fit_start_time
		
		# Drawing fitted histogram (if applicable)
		bkg_only_tf1=TF1("bkg_only_tf1","[0]*TMath::Exp([2]*(x-[1])+[3]*(x-[1])*(x-[1])+[4]*(x-[1])*(x-[1])*(x-[1]))",self.fitRangeLo,self.fitRangeHi,5)
		for i in range(self.NUM_PARMS_SIG,self.NUM_PARMS_SIG+self.NUM_PARMS_BKG): bkg_only_tf1.SetParameter(i-self.NUM_PARMS_SIG,tf1.GetParameter(i))
		bkg_only_Yield = bkg_only_tf1.Integral(self.sigIntegrateRangeLo,self.sigIntegrateRangeHi)/h.GetBinWidth(0)
		if(self.SaveCanvasLvl==2 or (self.SaveCanvasLvl==1 and var_tag=="nominal")):
			ctmp = TCanvas("ctmp","ctmp",1200,900)
			h.Draw() # Just to get stats box
			# Draw shaded box indicating bkg estimated yield
			h.SetAxisRange(self.fitRangeLo-0.05,self.fitRangeHi+0.05)
			max_y_val = h.GetBinContent(h.GetMaximumBin())
			h.GetYaxis().SetRangeUser(0,max_y_val*1.2)
			title,xtitle,ytitle = "",self.xaxis_title,"Counts / 2 MeV" # Hist dictionary referenced contains top, x, and y axes; use only x-axis here
			if(h.GetNbinsX()!=500): ytitle="Counts / bin"
			h.SetTitle(title+";"+xtitle+";"+ytitle)
			h.GetYaxis().SetTitleOffset(1.4)
			# Move stats box slightly
			gStyle.SetOptFit(111)
			from ROOT import TPaveStats
			st = TPaveStats()
			st = h.FindObject("stats")
			if(st==None):
				if "bggen" not in var_tag: print("WARNING: TPaveStats not found!")
				# sys.exit()
			else:
				st.SetX1NDC(0.65)
				st.SetX2NDC(0.95)
				st.SetY1NDC(0.65)
				st.SetY2NDC(0.95)
			# # Draw based on fit case
			if(case_str=="FitBkgRegion"):
				left_region_tf1 = tf1.Clone()
				left_region_tf1.SetRange(self.fitRangeLo,tf1.GetParameter(self.NUM_PARMS_SIG+self.NUM_PARMS_BKG) )
				left_region_tf1.Draw("same")
				center_region_tf1 = tf1.Clone()
				center_region_tf1.SetFillColor(kRed)
				center_region_tf1.SetFillStyle(3144)
				center_region_tf1.SetRange(tf1.GetParameter(self.NUM_PARMS_SIG+self.NUM_PARMS_BKG),tf1.GetParameter(self.NUM_PARMS_SIG+self.NUM_PARMS_BKG+1) )
				center_region_tf1.Draw("same")
				right_region_tf1 = tf1.Clone()
				right_region_tf1.SetRange(tf1.GetParameter(self.NUM_PARMS_SIG+self.NUM_PARMS_BKG+1),self.fitRangeHi )
				right_region_tf1.Draw("same")
				# ctmp.SaveAs(self.FITTED_HIST_PLOTDIR+self.canvas_tag+"_"+h.GetName()[2:]+"_"+var_tag+"_Case1.pdf")
				with suppress_stdout_stderr(): ctmp.SaveAs(self.FITTED_HIST_PLOTDIR+self.canvas_tag+"_"+h.GetName()[2:]+"_"+var_tag+"_Case1.pdf")
			# Draw fit function only in signal range
			elif(case_str=="FitSigRegion"):
				center_region_tf1 = tf1.Clone()
				center_region_tf1.SetRange(tf1.GetParameter(self.NUM_PARMS_SIG+self.NUM_PARMS_BKG),tf1.GetParameter(self.NUM_PARMS_SIG+self.NUM_PARMS_BKG+1) )
				center_region_tf1.Draw("same")
				bkg_only_tf1.SetLineWidth(1)
				# bkg_only_tf1.SetLineColor(kBlack)
				bkg_only_tf1.SetLineStyle(kDashed)
				bkg_only_tf1.Draw("same")
				# ctmp.SaveAs(self.FITTED_HIST_PLOTDIR+self.canvas_tag+"_"+h.GetName()[2:]+"_"+var_tag+"_Case2.pdf")
				with suppress_stdout_stderr(): ctmp.SaveAs(self.FITTED_HIST_PLOTDIR+self.canvas_tag+"_"+h.GetName()[2:]+"_"+var_tag+"_Case2.pdf")
			# Draw fit function in wide range
			elif(case_str=="FitFullRegion"):
				full_region_tf1 = tf1.Clone()
				full_region_tf1.SetRange(self.fitRangeLo,self.fitRangeHi )
				full_region_tf1.Draw("same")
				bkg_only_tf1.SetLineWidth(1)
				# bkg_only_tf1.SetLineColor(kBlack)
				bkg_only_tf1.SetLineStyle(kDashed)
				bkg_only_tf1.Draw("same")
				# ctmp.SaveAs(self.FITTED_HIST_PLOTDIR+self.canvas_tag+"_"+h.GetName()[2:]+"_"+var_tag+"_Case3.pdf")
				with suppress_stdout_stderr(): ctmp.SaveAs(self.FITTED_HIST_PLOTDIR+self.canvas_tag+"_"+h.GetName()[2:]+"_"+var_tag+"_Case3.pdf")
			del ctmp
			
		# Output stuff
		if(case_str=="FitFullRegion"): 
			return int(fit_result),chi2Uncert,bkg_only_Yield
		else: return int(fit_result),chi2Uncert

	# Return our tf1 function based on settings from parm_list
	# Parm list of lists: each inner list is of form ["string name",init,min,max,fix=True/False]
	# If NumParmsToSet =0 (default) then check against total number of paramters for sig + bkg
	# If NumParmsToSet!=0 then check that the number of paramters expected and provided match
	def jz_FitFunctionSetup(self,tf1,parm_list,NumParmsToSet=0):
		
		from ROOT import jzTwoGausSig_ExpBkg
		
		# Initial setup of fit functions
		if(tf1==None):
			tf1 = TF1("my_tot_fit",jzTwoGausSig_ExpBkg,self.fitRangeLo,self.fitRangeHi,self.NUM_PARMS_SIG+self.NUM_PARMS_BKG+3) #Why +3? Last three parms are sig range min,max,whether to reject sig/bkg region
			tf1.SetNpx(1000) # smoother line
		
		if(NumParmsToSet==0 and len(parm_list)!=self.NUM_PARMS_SIG+self.NUM_PARMS_BKG):
			print("ERROR: number of paramters found in list does not match number of sig+bkg!")
			print("parameter list: ")
			for list in parm_list:
				print(str(list))
			print("Number of expected from sig+bkg: " + str(self.NUM_PARMS_SIG+self.NUM_PARMS_BKG) + " Num found: " + str(len(parm_list)))
			sys.exit()
		if(NumParmsToSet!=0 and len(parm_list)!=NumParmsToSet):
			print("ERROR: incorrect number of paramters found in list does not match expectations!")
			sys.exit()
		
		for i in range(0,len(parm_list)):
			singleParmList = parm_list[i]
			tf1.SetParName(i,singleParmList[0])
			FixThisParm = singleParmList[4]
			# print "Setting up for parm: " + singleParmList[0]
			# print "Fix parm? " + str(FixThisParm)
			if(FixThisParm):
				tf1.FixParameter(i,singleParmList[1])
				# print "This parm fixed: " + singleParmList[0]
			else:
				tf1.SetParameter(i,singleParmList[1])
				tf1.SetParLimits(i,singleParmList[2],singleParmList[3])
			
		# Add signal range min + max as two additional (fixed) parameters at the endswith
		tf1.FixParameter(self.NUM_PARMS_SIG+self.NUM_PARMS_BKG,self.sigRejectRangeLo)
		tf1.FixParameter(self.NUM_PARMS_SIG+self.NUM_PARMS_BKG+1,self.sigRejectRangeHi)
			
		tf1.SetNpx(1000) # smoother line
			
		return tf1

	
	def SetNegativeBinsToZero(self,h):
		old_integral=h.Integral()
		for i in range(0,h.GetNbinsX()+2): 
			if(h.GetBinContent(i)<0.):
				h.SetBinContent(i,0.)
				h.SetBinError(i,0.)
		new_integral=h.Integral()
		return
		
	def jzHistogramSumIntegral(self,h,min_hist_val=-1e9,max_hist_val=1e9):
		sum = 0.0
		for i in range(0,h.GetNbinsX()+2): 
			if(h.GetBinLowEdge(i)+1e-4>=min_hist_val and (h.GetBinLowEdge(i)+h.GetBinWidth(1))<=max_hist_val+1e-4): sum+=h.GetBinContent(i) #Recall: 0 is underflow, nbinsx is maximum bin, nbinsx+1 is overflow bin. I think.
		return sum
		
	def jzHistogramErrSumIntegral(self,h,min_hist_val=-1e9,max_hist_val=1e9):
		sum2 = 0.0
		for i in range(0,h.GetNbinsX()+2): 
			if(h.GetBinLowEdge(i)+1e-4>=min_hist_val and (h.GetBinLowEdge(i)+h.GetBinWidth(1))<=max_hist_val+1e-4): sum2+=h.GetBinError(i)**2 #Recall: 0 is underflow, nbinsx is maximum bin, nbinsx+1 is overflow bin. I think.
		return sqrt(sum2)
		

# Define a context manager to suppress stdout and stderr.
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      
    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


#####################################################
###### Things from my pyroot helper file ############
#####################################################


# shell_exec(command, stdin_str):
# # command is the string (space separated)

# returns a numpy array for whichever axis (or axis uncertainty) is specified
# str_which_array options are "X", "EX", "Y", or "EY", for whichever I want
# jz_tgraph2numpy(gr,str_which_array):
# jz_th1f2numpy(gr,str_which_array)   
# jzGetRandomColor()   
# jz_DressUpObject(tobj,NewNameStr,kColor=kBlack,kMarkerStyle=kFullCircle,title="",xtitle="",ytitle="",Opacity=1.0,MarkerSize=1.0):
# jz_get_hist_binnum(my_val,nbins,h_min,h_max) 
# jzPlotHistStack(h_list,legend_list,tag_name,rebin_factor=1,range_lo=-1000,range_hi=1000,legend_limits=[],SavePNG=True):


#####################################################
###### Simple PyRoot Examples            ############
#####################################################

# Some example code...

## Opening a TFile (read-only)
# f = TFile.Open("name.root")
## Creating a new TFile
# f = TFile.Open("name.root","RECREATE")
## Modifying an existing file 
# f = TFile.Open("name.root","UPDATE")

## Retrieving TObject from TFile
# h   = f.Get("hname")

## Arrays
# my_arr = array('d',[])

## TLegend
# legend = TLegend(xmin,ymin,xmax,ymax) #0.1 is lower limit of plot, 0.9 is upper limit (beyond on either side is labeling+whitespace)
# legend.AddEntry(h,"Label","pl")
# legend.Draw()

## Marking up a histogram for plotting
# def AddPlotCosmetics(fname,gr_name_str,kColor,kMarkerStyle):
## Need to return to this one...

## Saving a drawn histogram/graph/whatever
# c1.SaveAs("CompareSigmas.png") # .png, .pdf, .C, ...





