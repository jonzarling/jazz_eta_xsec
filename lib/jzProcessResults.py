#!/usr/bin/env python3

from optparse import OptionParser
import os.path
import os
import sys
import subprocess
import glob
from array import array
from math import sqrt, exp, sin, cos, asin, acos

#Root stuff
from ROOT import TFile, TTree, TBranch, TLorentzVector, TLorentzRotation, TVector3
from ROOT import TCanvas, TMath, TH2F, TH1F, TRandom, TGraphErrors, TPad, TGraphAsymmErrors, TGraph, TLine, TLegend, TLatex
from ROOT import gBenchmark, gDirectory, gROOT, gStyle, gPad, gSystem, SetOwnership

#Other packages
# import uproot
import numpy as np
import pandas as pd

#My Stuff
#Required: add to PYTHONPATH environment variable, e.g.
from jzEtaMesonXSec    import *
from jzPlottingScripts import *

class jzProcessResults():

	# Initialize function 
	def __init__(self):

		# For convenience
		self.jzEtaMesonXSec = jzEtaMesonXSec("gg","SP17","nominal") # Dummy values, use gg for widest binning, rest not important
		# This object holds all graphs and graphing options
		self.jzEtaXSecPlotter = jzEtaXSecPlotter(self.jzEtaMesonXSec.NOTE_PLOTS_DIR,jzEtaMesonXSec.THEORY_DIR,jzEtaMesonXSec.EXPT_DIR,self.jzEtaMesonXSec.UCHANNEL_TMIN_DIVIDER,self.jzEtaMesonXSec.UCHANNEL_TMAX_DIVIDER)
		self.df_dict = {}
		self.TEX_RESULT_DIR=self.jzEtaMesonXSec.TEX_RESULT_DIR
		self.NUM_E_BINS = 10
		self.NUM_W_BINS = 23
		self.EGAM_TABLE = [
			"$6.5<E_\gamma<7.0$",
			"$7.0<E_\gamma<7.5$",
			"$7.5<E_\gamma<8.0$",
			"$8.0<E_\gamma<8.5$",
			"$8.5<E_\gamma<9.0$",
			"$9.0<E_\gamma<9.5$",
			"$9.5<E_\gamma<10.0$",
			"$10.0<E_\gamma<10.5$",
			"$10.5<E_\gamma<11.0$",
			"$11.0<E_\gamma<11.5$",
			"$11.5<E_\gamma<12.0$",
			]

	# Assumes the run and mode will be somewhere in the tagname
	# # INTERNAL UNIQUE NAME: [mode]_[run]_[tag]
	def AddDF(self,mode,run,tag,fname_alt=""):
		
		fname="" if fname_alt=="" else self.jzEtaMesonXSec.XSEC_RESULT_DIR + fname_alt			
		
		# Get file
		if(fname==""):
			fname = self.jzEtaMesonXSec.XSEC_RESULT_DIR + mode + "_" + run + "_" + tag + "_RESULTS_ebin0to_ebin9.csv"
			if(mode=="COMBINED" or run=="COMBINED"): fname = self.jzEtaMesonXSec.XSEC_RESULT_DIR + "COMBINED_RESULTS_ebin0to_ebin9.csv"
			if(run=="FA18LE" and mode!="COMBINED"):  fname = self.jzEtaMesonXSec.XSEC_RESULT_DIR + mode + "_" + run + "_" + tag + "_RESULTS_ebin0to_ebin22.csv"
			if(run=="FA18LE" and mode=="COMBINED"):  fname = self.jzEtaMesonXSec.XSEC_RESULT_DIR + "COMBINED_RESULTS_LE_ebin0to_ebin22.csv"
		
		if(not os.path.exists(fname)):
			print(("WARNING: DF file not found: " + fname))
			return
			# sys.exit()
		# Add to dictionary
		df_key = mode+"_"+run+"_"+tag
		df = pd.read_csv(fname,index_col="totbin")

		if(df_key in self.df_dict):
			print(("DF already added, skipping... "+df_key))
			return

		# Hacky fixes for error bars
		if (mode != "COMBINED" and run != "COMBINED"):
			df.loc[ np.abs(df["xsec_SystErrLo_fit"]) > 50., "xsec_SystErrLo_fit"] = 0
			df.loc[ np.abs(df["xsec_SystErrHi_fit"]) > 50., "xsec_SystErrHi_fit"] = 0

		# Create new column of fit + stat err
		if("xsec_SystErrLo_fit" in df):
			df["xsec_SystErrLo_fit"] = df["xsec_SystErrLo_fit"].abs()
			df["xsec_SystErrFit"]    = df[["xsec_SystErrLo_fit", "xsec_SystErrHi_fit"]].max(axis=1)
			df["xsec_FitAndStatErr"] = np.sqrt(df["xsec_SystErrFit"]**2.+df["xsec_StatErr"]**2.)
		elif(mode=="COMBINED" and run=="COMBINED"):
			df["systFitLo_tot"] = df["systFitLo_tot"].abs()
			df["xsec_SystErrFit"]    = df[["systFitLo_tot", "systFitHi_tot"]].max(axis=1)
			df["xsec_FitAndStatErr"] = np.sqrt(df["xsec_SystErrFit"]**2.+df["statErr_tot"]**2.)
		elif(mode!="COMBINED" and run=="COMBINED"):
			df["systFitLo_tot"] = df["systFitLo_tot"].abs()
			df["xsec_SystErrFit"]    = df[["systFitLo_tot", "systFitHi_tot"]].max(axis=1)
			df["xsec_FitAndStatErr"] = np.sqrt(df["xsec_SystErrFit"]**2.+df["statErr_tot"]**2.)
		elif("totErrLo_tot" not in df):
		# else:
			print("ERROR in loading DF, exiting...")
			sys.exit()

		self.df_dict[df_key] = df

	def DFName(self,mode,run,tag): return mode+"_"+run+"_"+tag

	# Create graph of some variable (xsec by default) over |t|
	# # INTERNAL UNIQUE NAME: [mode]_[run]_[tag]_[yvar]_[ebin]
	# # ERROR BARS:
	# # # If yvar_E=="",          no errors
	def CreateGraphSingleDF(self,mode,run,tag,ebin,yvar="xsec",yvar_Err="",yaxis_title="",legend=""):
		
		PlotNoError    = True if yvar_Err == "" else False
		isLowE         = True if(run == "FA18LE")  else False
		
		gr_name = mode+"_"+run+"_"+tag+"_"+yvar+"_"+str(ebin)
		yvar_key = yvar
		
		# Override when dealing with combined results (over runs/modes)
		if(run=="COMBINED"):
			if(yvar=="xsec"):
				if(mode=="COMBINED"): yvar_key = "xsec_tot"
				else:                 yvar_key = "xsec_Wavg_"+mode
			elif(yvar=="xsec_ExclIncl_tot"): yvar_key = "xsec_ExclIncl_tot"
			else:
				print(("ERROR, yvar " + yvar + " not supported for COMBINED yet, exiting..."))
				sys.exit()
		if(run=="FA18LE" and mode=="COMBINED"): yvar_key = "xsec_tot"
		
		# Check that df is registered, create a tmp copy
		df_key = mode+"_"+run+"_"+tag
		if(df_key not in self.df_dict): 
			print(("WARNING: could not find key " + df_key + " in dataframes dict! All keys:"))
			return
			# for key in self.df_dict: print "key: " + key
			# sys.exit()
		df = self.df_dict[df_key].copy()
		# Get correct ebin/wbin ONLY
		df=df[df["ebin"] == ebin].copy()
		
		# Check that all variables can be found
		if(yvar_key not in df):
			print(("ERROR: y-axis variable not found for df " + df_key + " exiting..."))
			print(("yvar_key: " + yvar_key))
			print(("Graphname: " + gr_name))
			sys.exit()
		if(not PlotNoError and yvar_Err not in df):
			print(("ERROR: y-axis variable not found for df " + df_key))
			print(("Variable: " + yvar_Err))
			sys.exit()
				
		# Retrieve graph values from df
		nbins = len(df)
		tbins_np        = (jz_DF_to_numpy(df,"tbinLo")+jz_DF_to_numpy(df,"tbinHi"))/2.
		tbinsErr_np     = np.zeros(nbins)
		y_np            = jz_DF_to_numpy(df,yvar_key)
		yErr            = jz_DF_to_numpy(df,yvar_Err) if not PlotNoError else np.zeros(nbins)

		# For first encounter of standard E runs, add t-range to plotter class
		if(not isLowE and self.jzEtaXSecPlotter.xaxis_range[ebin].shape[0]==0): self.jzEtaXSecPlotter.xaxis_range[ebin]=tbins_np
		if(isLowE     and self.jzEtaXSecPlotter.xaxis_rangeLE.shape[0]==0):     self.jzEtaXSecPlotter.xaxis_rangeLE=tbins_np




		# if(mode=="gg" and run=="COMBINED"):
		# 	print("y-vals: "+str(y_np))
		# 	print("y-vals yvar_key: "+str(yvar_key))
		# 	sys.exit()

		# Create TGraphErrors
		gr = TGraphErrors(nbins,tbins_np,y_np,tbinsErr_np,yErr)
		if(yaxis_title!=""): gr.GetYaxis().SetTitle(yaxis_title)
		self.jzEtaXSecPlotter.AddGraph(gr,gr_name,mode,run,ebin,yvar=yvar,legend=legend)
		
		del df
		return

	def PlotMultiResults5x2(self,plot_name,modes,runs,tags,yvars=["xsec"],RegionCase=0,drawLegend=False,drawLogY=False,graphMax=-1.,FitDrawPol0=False):
		self.jzEtaXSecPlotter.PlotMultiResults5x2(plot_name,modes,runs,tags,yvars,RegionCase,drawLegend,drawLogY,graphMax,FitDrawPol0)
	def PlotMultiResultsOneEbin(self,plot_name,modes,runs,tags,yvars=["xsec"],ebin=0,RegionCase=0,multipad=False,drawLegend=False,drawLogY=False,canvasBin=-1,FitDrawPol0=False,legend_tuple=(),graph_xmin_override=-1.,graph_xmax_override=-1.,graph_ymin_override=-1.,graph_ymax_override=-1.):
		self.jzEtaXSecPlotter.PlotMultiResultsOneEbin(plot_name,modes,runs,tags,yvars,ebin,RegionCase,multipad,drawLegend,drawLogY,canvasBin,FitDrawPol0,legend_tuple,graph_xmin_override,graph_xmax_override,graph_ymin_override,graph_ymax_override)
	def PlotMultiResultsLE(self,plot_name,modes,runs,tags,yvars=["xsec"],drawLegend=False,graphMax=-1.,drawLogY=False):
		self.jzEtaXSecPlotter.PlotMultiResultsLE(plot_name,modes,runs,tags,yvars,drawLegend,drawLogY)
	def PlotGlueX_4GeV_Comparison(self,df_key="COMBINED_FA18LE_nominal"):
		self.jzEtaXSecPlotter.PlotGlueX_4GeV_Comparison(df=self.df_dict[df_key])
	def PlotGlueX_8GeV_Comparison(self,df_key="COMBINED_nominal"):
		self.jzEtaXSecPlotter.PlotGlueX_4GeV_Comparison(df=self.df_dict[df_key])

	def CreateCombinedDF(self, cutoff, is_LE=False, plotBGGEN=True):


		# Results used to calculate total combined DF
		modes = ["gg", "3pi0", "3piq"]
		runs = ["SP17", "SP18", "FA18"]
		variations = ["nominal"]
		variations += ["FidInnerCut_" + str(val) for val in [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]]
		if(is_LE):
			runs = ["FA18LE"]
			modes = ["gg"]
			self.jzEtaMesonXSec.XSEC_RESULT_DIR = "/w/halld-scshelf2101/home/jzarling/jzXSecTools/xsec_results/"
			# self.jzEtaMesonXSec.XSEC_RESULT_DIR = "/w/halld-scshelf2101/home/jzarling/OLDjzXSecTools/xsec_results/"
		# Import results
		for mode in modes:
			for run in runs:
				for var in variations:
					self.AddDF(mode,run,var)
		if(is_LE): self.jzEtaMesonXSec.XSEC_RESULT_DIR = "/w/halld-scshelf2101/home/jzarling/jzXSecTools/xsec_results/"


		# Components of uncertainty:
		# # Statistical
		# # Fit uncert. from max(run_period_syst)
		# # rms[gg run periods], weighted by stats
		# # rms(gg fiducial volume results), individually for each run period, weighted by stats, MAX(run_periods)
		# # rms(gg - 3pi0 results), weighted by stat+run+fit uncert.
		# # Single particle uncertainty
		# # BGGEN
		# # Flux + proton (not calculated here)

		# COMBINED DF COLUMNS
		init_cols = ["ebin","tbin","tbinLo","tbinHi"]
		df_tagname = "gg_SP17_nominal" if not is_LE else "gg_FA18LE_nominal"
		# Initialize output dataframe from some (randomly picked) input df
		df_out = pd.DataFrame(data=self.df_dict[df_tagname][init_cols],index=self.df_dict[df_tagname].index,copy=True)
		for col in ["xsec_tot","xsec_ExclIncl_tot","xsec_Incl_tot","totErrLo_tot","totErrHi_tot","systErrLo_tot","systErrHi_tot","statErr_tot","systFidInner_tot",
					"systFitLo_tot","systFitHi_tot","systRuns_tot","systModes_tot","systSinglePart_xsec_tot"]:
			df_out[col]=0.
		df_out["divider"]="*"

		# COMBINED DF COLUMNS FOR INDIVIDUAL MODES
		for mode in modes:
			df_out["dividerSTART_" + mode] = "*"
			for col in ["xsec_Wavg_"+mode,"NSig_"+mode+"_SP17","NSig_"+mode+"_SP18","NSig_"+mode+"_FA18","totErrLo_Wavg_"+mode,"totErrHi_Wavg_"+mode,
						"statErr_Wavg_"+mode,"systFidInner_Wavg_"+mode,"systFitLo_Wavg_"+mode,"systFitHi_Wavg_"+mode,"systRuns_Wavg_"+mode,"systNeutSP_frac_"+mode,
						"systNeutSP_globfrac_"+mode,"systNeutSP_ptp_"+mode,"NFCAL_avg_"+mode,"NBCAL_avg_"+mode,
						"xsec_conv_fact_SP17_"+mode,"xsec_conv_fact_SP18_"+mode,"xsec_conv_fact_FA18_"+mode,]:
				df_out[col] = 0.
			df_out["dividerEND_"+mode] = "*"

		for mode in modes:
			for run in runs:
				df_out["NSig_"+mode+"_"+run]=self.df_dict[mode+"_"+run+"_nominal"]["avgSigYield"]
				df_out["xsec_conv_fact_"+run+"_"+mode] = self.df_dict[mode+"_"+run+"_nominal"]["xsec"] / self.df_dict[mode+"_"+run+"_nominal"]["avgSigYield"]

		if(is_LE): df_out.drop(columns=["systRuns_tot","systModes_tot","systRuns_Wavg_gg",],inplace=True)

		# Calculate uncertainty from variations within single mode+run
		for mode in modes:
			for run in runs:
				self.CalcUncertSingleResult(mode,run,variations,df_out)
			if(not is_LE): df_out["systFidInner_Wavg_"+mode]=df_out[["systFidInner_Wavg_"+mode+"_SP17","systFidInner_Wavg_"+mode+"_SP18","systFidInner_Wavg_"+mode+"_FA18",]].max(axis=1)
			if(is_LE):     df_out["systFidInner_Wavg_"+mode]=df_out["systFidInner_Wavg_"+mode+"_FA18LE"]
			df_out["divider_"+mode] = "*"
		# Get combined fit syst (max over 3 runs) and avg. num photons
		if(is_LE): df_out["NFCAL_avg_" + mode], df_out["NBCAL_avg_" + mode] = self.df_dict[df_tagname]["NFCAL_avg"],self.df_dict[df_tagname]["NBCAL_avg"]
		if(not is_LE):
			for mode in modes:
				df_out["NFCAL_avg_" + mode], df_out["NBCAL_avg_" + mode] = 0., 0.
				for run in runs:
					df_name=mode+"_"+run+"_nominal"
					df_out["NFCAL_avg_"+mode]+=self.df_dict[df_name]["NFCAL_avg"]/3.
					df_out["NBCAL_avg_"+mode]+=self.df_dict[df_name]["NBCAL_avg"]/3.

		# COMBINE RESULTS OVER 3 RUN PERIODS! KEEP INDIVIDUAL MODES FOR NOW
		# # Recall that dfs are named by string [mode]_[run]_[tag]
		if(not is_LE):
			for mode in modes: self.CombineSingleModeOverRuns(mode,df_out,cutoff)
		df_out.replace([np.nan, np.inf, -np.inf], 0., inplace=True) # Remove nan and infs

		# Add neutral per-particle uncert., split into global and point-to-point components (gg only)
		for mode in ["gg"]:
			print("WARNING, ONLY CALCULATING PER PARTICLE UNCERT ON GG MODE")
			df_out["systNeutSP_frac_"+mode] = ((df_out["NFCAL_avg_"+mode]*0.02)**2. + (df_out["NBCAL_avg_"+mode]*0.05)**2.)**0.5
			minNeutralSP=df_out["systNeutSP_frac_"+mode].min()
			df_out["systNeutSP_globfrac_"+mode]=minNeutralSP
			# df_out["systNeutSP_ptp_"+mode] = (df_out["xsec_Wavg_gg"] * (df_out["systNeutSP_frac_"+mode]**2.) -(df_out["systNeutSP_globfrac_"+mode]**2.) )**0.5
			if(not is_LE): df_out["systNeutSP_ptp_"+mode] = df_out["xsec_Wavg_gg"] * df_out["systNeutSP_frac_"+mode]
			if(is_LE):     df_out["systNeutSP_ptp_"+mode] = self.df_dict[df_tagname]["xsec"] * df_out["systNeutSP_frac_"+mode]


		# Copy gg results to "combined", get BGGEN
		if(not is_LE):
			# Add BGGEN uncertainty component (in colname "xsec_Incl_tot")
			self.CalcBGGENUncert(df_out,plotHists=plotBGGEN)
			df_out["xsec_ExclIncl_tot"]  = df_out["xsec_Wavg_gg"]
			df_out["statErr_tot"]        = df_out["statErr_Wavg_gg"]
			df_out["systFidInner_tot"]   = df_out["systFidInner_Wavg_gg"]
			df_out["systFitLo_tot"]      = df_out["systFitLo_Wavg_gg"]
			df_out["systFitHi_tot"]      = df_out["systFitHi_Wavg_gg"]
			df_out["systRuns_tot"]       = df_out["systRuns_Wavg_gg"]
			df_out["systSinglePart_xsec_tot"] = df_out["systNeutSP_ptp_gg"]
		# Copy gg results to "combined"
		if(is_LE):
			df_out["xsec_ExclIncl_tot"]           = self.df_dict[df_tagname]["xsec"]
			df_out["statErr_tot"]        = self.df_dict[df_tagname]["xsec_StatErr"]
			df_out["systFidInner_tot"]   = df_out["systFidInner_Wavg_gg"]
			df_out["systFitLo_tot"]      = self.df_dict[df_tagname]["xsec_SystErrLo_fit"]
			df_out["systFitHi_tot"]      = self.df_dict[df_tagname]["xsec_SystErrHi_fit"]
			df_out["systSinglePart_xsec_tot"] = df_out["systNeutSP_ptp_gg"]


		# Hacky fixes before we plot
		df_out.loc[df_out["systFitLo_tot"] > 150., "systFitLo_tot"] = 0


		if not is_LE:
			df_out["xsec_tot"] = df_out["xsec_ExclIncl_tot"]-df_out["xsec_Incl_tot"]
			# Total uncert. for each mode
			componentsLo = ["statErr_Wavg_", "systFidInner_Wavg_", "systFitLo_Wavg_", "systRuns_Wavg_","systNeutSP_ptp_"]
			componentsHi = ["statErr_Wavg_", "systFidInner_Wavg_", "systFitHi_Wavg_", "systRuns_Wavg_","systNeutSP_ptp_"]
			for mode in modes:
				df_out["totErrLo_Wavg_"+mode], df_out["totErrHi_Wavg_"+mode] = 0., 0.
				for v in componentsLo: df_out["totErrLo_Wavg_"+mode] += df_out[v+mode]**2.
				for v in componentsHi: df_out["totErrHi_Wavg_"+mode] += df_out[v+mode]**2.
				df_out["totErrLo_Wavg_"+mode]=df_out["totErrLo_Wavg_"+mode]**0.5
				df_out["totErrHi_Wavg_"+mode]=df_out["totErrHi_Wavg_"+mode]**0.5
			self.CalcModeDiffUncert(df_out) # Fills out df_out["systModes_tot"] Get gg vs. 3pi0 uncertainty

			# Propogate gg results to total & add a few more components
			# # (this is also where we add bggen, flux, and mode uncert.)
			df_out["totErrLo_tot"] = np.sqrt(df_out["totErrLo_Wavg_gg"] ** 2. + df_out["systModes_tot"] ** 2. + df_out["xsec_Incl_tot"] ** 2.)
			df_out["totErrHi_tot"] = np.sqrt(df_out["totErrHi_Wavg_gg"] ** 2. + df_out["systModes_tot"] ** 2. + df_out["xsec_Incl_tot"] ** 2.)
		if is_LE:
			df_out["xsec_tot"] = df_out["xsec_ExclIncl_tot"]
			df_out["totErrLo_tot"] = np.sqrt(df_out["statErr_tot"]**2. + df_out["systFidInner_tot"]**2. + df_out["systSinglePart_xsec_tot"]**2. + df_out["systFitLo_tot"]**2.)
			df_out["totErrHi_tot"] = np.sqrt(df_out["statErr_tot"]**2. + df_out["systFidInner_tot"]**2. + df_out["systSinglePart_xsec_tot"]**2. + df_out["systFitHi_tot"]**2.)
		df_out["systErrLo_tot"] = np.sqrt(df_out["totErrLo_tot"]**2. - df_out["statErr_tot"]**2.)
		df_out["systErrHi_tot"] = np.sqrt(df_out["totErrHi_tot"]**2. - df_out["statErr_tot"]**2.)


		df_out_name = self.jzEtaMesonXSec.XSEC_RESULT_DIR + "COMBINED_RESULTS_ebin0to_ebin9.csv" if not is_LE else self.jzEtaMesonXSec.XSEC_RESULT_DIR + "COMBINED_RESULTS_LE_ebin0to_ebin22.csv"
		print(("Saving combined file: " + df_out_name))
		df_out.to_csv(df_out_name)

	# Calculate weights (based on stats. uncert)
	def CombineSingleModeOverRuns(self, mode, df_out, cutoff):

		dfs_run_periods = [mode + "_SP17_nominal", mode + "_SP18_nominal", mode + "_FA18_nominal"]
		quantities_to_prop = ["xsec", "xsec_SystErrLo_fit",
							  "xsec_SystErrHi_fit"]  # Quantities that add linearly (not in quad.)

		# First need sum of weights in order to calculate normalized weights
		# # Also, skip zero yields. df[WUnnorm_sum_"+mode]=0 is equivalent to skipping
		# # WEIGHTS WILL NORMALIZE FINE, EVEN WHEN A RUN PERIOD FALLS BELOW CUTOFF. HAKUNA MATATA.
		df_out["WUnnorm_sum_" + mode] = 0.
		for df_name in dfs_run_periods:
			NSig_key = "NSig_" + df_name.split("_nominal")[0]  # e.g. "NSig_gg_SP17"
			df_out["WUnnorm_" + mode + "_" + df_name] = 0.
			df_out.loc[df_out[NSig_key] > cutoff, "WUnnorm_" + mode + "_" + df_name] = self.df_dict[df_name]["xsec_StatErr"] ** -2.  # If yield > cutoff, store weight, otherwise weight stays 0
			df_out["WUnnorm_sum_" + mode] += df_out["WUnnorm_" + mode + "_" + df_name]

		# Initialize output columns, for quantities that don't exist yet
		df_out["xsec_Wavg_" + mode] = 0.
		df_out["xsec_SystErrLo_fit_Wavg_" + mode] = 0.
		df_out["xsec_SystErrHi_fit_Wavg_" + mode] = 0.
		df_out["statErr_Wavg_" + mode] = 0.  # Separate, since it adds in quad.

		# Now get normalized weights and weighted values of xsec + related quantities
		for q in quantities_to_prop: df_out[q + "_Wavg_" + mode] = 0.
		for df_name in dfs_run_periods:
			df_out["W_" + df_name] = df_out["WUnnorm_" + mode + "_" + df_name] / df_out["WUnnorm_sum_" + mode]
			for q in quantities_to_prop: df_out[q + "_Wavg_" + mode] += df_out["W_" + df_name] * self.df_dict[df_name][
				q]  # Quantities that add linearly
			df_out["statErr_Wavg_" + mode] += (df_out["W_" + df_name] * self.df_dict[df_name][
				"xsec_StatErr"]) ** 2.  # Add in quad (THEN TAKE SQRT BELOW)
		df_out["statErr_Wavg_" + mode] = df_out["statErr_Wavg_" + mode] ** 0.5  # finally, take sqrt of sum-in-quad

		# Add sample standard deviation (weighted) of results over 3 runs
		n = len(dfs_run_periods)
		df_out["systRuns_Wavg_" + mode] = 0.
		for df_name in dfs_run_periods:
			df_out["systRuns_Wavg_" + mode] += (df_out["W_" + df_name]) * (
						self.df_dict[df_name]["xsec"] - df_out["xsec_Wavg_" + mode]) ** 2.
		df_out["systRuns_Wavg_" + mode] = (df_out["systRuns_Wavg_" + mode] / ((n - 1.) / n)) ** (0.5)

		# Renamed columns (difference between naming convention here and when creating df from fitting step)
		df_out["systFitLo_Wavg_" + mode] = df_out["xsec_SystErrLo_fit_Wavg_" + mode]
		df_out["systFitHi_Wavg_" + mode] = df_out["xsec_SystErrHi_fit_Wavg_" + mode]

		# For reference, store individual run results
		for q in quantities_to_prop:
			for df_name in dfs_run_periods:
				df_out[q + "_" + df_name] = self.df_dict[df_name][q]

	# Calculate weights (based on stats. uncert)
	def CalcUncertSingleResult(self, mode, run, var_list, df_out):
		curr_tag = "systFidInner_" + mode + "_" + run

		# First need sum of weights in order to calculate normalized weights
		df_out["WUnnorm_sum_" + curr_tag] = 0.
		for var in var_list:
			df_name = mode + "_" + run + "_" + var
			df_out["WUnnorm_" + curr_tag + "_" + var] = self.df_dict[df_name]["xsec_StatErr"] ** -2.
			df_out["WUnnorm_sum_" + curr_tag] += self.df_dict[df_name]["xsec_StatErr"] ** -2.
		# Initialize output columns, for quantities that don't exist yet
		df_out["xsec_Wavg_systFidInner" + mode + "_" + run] = 0.
		# Now get normalized weights and weighted values of xsec + related quantities
		for var in var_list:
			df_name = mode + "_" + run + "_" + var
			df_out["W_" + curr_tag + "_" + var] = df_out["WUnnorm_" + curr_tag + "_" + var] / df_out["WUnnorm_sum_" + curr_tag]
			df_out["xsec_Wavg_systFidInner" + mode + "_" + run] += df_out["W_" + curr_tag + "_" + var] * self.df_dict[df_name]["xsec"]  # Calculate weight
		# Add sample standard deviation (weighted) of results over variations
		n = len(var_list)
		df_out["systFidInner_Wavg_" + mode + "_" + run] = 0.
		for var in var_list:
			df_name = mode + "_" + run + "_" + var
			df_out["systFidInner_Wavg_" + mode + "_" + run] += (df_out["W_" + curr_tag + "_" + var]) * (self.df_dict[df_name]["xsec"] - df_out["xsec_Wavg_systFidInner" + mode + "_" + run]) ** 2.
		df_out["systFidInner_Wavg_" + mode + "_" + run] = (df_out["systFidInner_Wavg_" + mode + "_" + run] / ((n - 1.) / n)) ** (0.5)

	def CalcModeDiffUncert(self, df_out):

		# First need sum of weights in order to calculate normalized weights
		# # use average of low/high uncert. here
		df_out["WUnnorm_MODES_gg"] = ((df_out["totErrLo_Wavg_gg"] + df_out["totErrHi_Wavg_gg"]) / 2.) ** -2.
		df_out["WUnnorm_MODES_3pi0"] = ((df_out["totErrLo_Wavg_3pi0"] + df_out["totErrHi_Wavg_3pi0"]) / 2.) ** -2.
		df_out["WUnnorm_sum_MODES"] = df_out["WUnnorm_MODES_gg"] + df_out["WUnnorm_MODES_3pi0"]

		# Now get normalized weights and weighted values of xsec + related quantities
		df_out["W_MODES_gg"] = df_out["WUnnorm_MODES_gg"] / df_out["WUnnorm_sum_MODES"]
		df_out["W_MODES_3pi0"] = df_out["WUnnorm_MODES_3pi0"] / df_out["WUnnorm_sum_MODES"]

		# Mode averaged value (only used for getting stdev
		df_out["xsec_Wavg_MODES"] = df_out["W_MODES_gg"] * df_out["xsec_Wavg_gg"] + df_out["W_MODES_3pi0"] * df_out[
			"xsec_Wavg_3pi0"]

		# Sample standard deviation (weighted) of results over 3 runs
		n = 2.
		df_out["systModes_tot"] = (df_out["W_MODES_gg"]) * (
					df_out["xsec_Wavg_gg"] - df_out["xsec_Wavg_MODES"]) ** 2.
		df_out["systModes_tot"] += (df_out["W_MODES_3pi0"]) * (
					df_out["xsec_Wavg_3pi0"] - df_out["xsec_Wavg_MODES"]) ** 2.
		df_out["systModes_tot"] = (df_out["systModes_tot"] / ((n - 1.) / n)) ** (0.5)

		# The WUnnorm_MODES_3pi0 will divide by zero and make NaNs, reset here
		df_out.replace([np.nan, np.inf, -np.inf], 0., inplace=True)  # Remove nan and infs

	def CalcBGGENUncert(self,comb_DF,data_hists="/w/halld-scshelf2101/home/jzarling/jzXSecTools/hist_files/nominal.root",plotHists=True):
		# Import if not already added
		if "gg_FA18_nominal" not in self.df_dict:
			print("Importing Fall 2018 DF for bggen hists...")
			self.AddDF("gg", "FA18", "nominal")

		ratio_arr = self.jzEtaXSecPlotter.CalcBGGENUncert(plotHists=plotHists)
		data_DF=self.df_dict["gg_FA18_nominal"]

		# print("ratio array: " + str(ratio_arr))

		f_data = TFile.Open(data_hists)
		for ebin in range(0,10):
			h2_data = f_data.Get("gg_FA18/h2_eta_kin_ebin"+str(ebin)+"_DATA")
			for tbin in range(1,20):
				totbin = ebin*100+tbin
				numSigBkg_data = h2_data.ProjectionX("h_eta_kin_ebin"+str(ebin)+"_tbin"+str(tbin), tbin+1, tbin+1).Integral()
				numBkg_data = numSigBkg_data - data_DF.loc[totbin]["avgSigYield"]
				numEstIncl  = ratio_arr[tbin]*numBkg_data
				# numEta  = data_DF.loc[totbin]["avgSigYield"]+numEstIncl
				comb_DF.at[totbin,"xsec_Incl_tot"] = comb_DF.loc[totbin]["xsec_conv_fact_FA18_gg"]*numEstIncl

	def SaveCombinedXSecToLatex(self,csv_file="xsec_results/COMBINED_RESULTS_ebin0to_ebin9.csv",systLoColName="systErrLo_tot",systHiColName="systErrHi_tot",saveLowE=False):
		df = pd.read_csv(csv_file,index_col="totbin")
		df["Syst_ptp"] = df[["systErrLo_tot", "systErrHi_tot"]].max(axis=1)
		with pd.option_context("max_colwidth", 1000):
			df["$|t|$ Range"]  = "$"+df["tbinLo"].round(2).astype(str)+"<|t|<"+df["tbinHi"].round(2).astype(str)+"$"
			# df["Syst_ptp"] = "\scriptsize{ ${\text{+"+df[systHiColName].round(3).astype(str)+"} \\atop \text{-"+df[systLoColName].round(3).astype(str)+"}}$ }"
			# df["Syst_HiLo"] = "\scriptsize{ ${\text{+"+df[systHiColName].round(3).astype(str)+"} \\atop \text{-"+df[systLoColName].round(3).astype(str)+"}}$ }"
			df["FluxUncert"] = "tbd" # Empy initially
			df["Beam Energy"] = "" # Empy initially
			# Beam energy column: only fill once per ebin (at FIRST entry)
			for ebin in range(0,self.NUM_E_BINS):
				df.at[ebin*100+1,"Beam Energy"] = self.EGAM_TABLE[ebin] # Text only in first row of table (1, 101, 201, etc --- recall that we skip tbin 0)
			# Now, split by energy bin to create tables that will fit on a single page
			for ebin in range(0,self.NUM_E_BINS):
				df_latexCols = df[df["ebin"]==ebin].copy()
				df_latexCols = df_latexCols[["Beam Energy","$|t|$ Range","xsec_tot","statErr_tot","Syst_ptp"]].copy()
				# df_latexCols = df_latexCols[["Beam Energy","$|t|$ Range","xsec_tot","statErr_tot","Syst_HiLo","FluxUncert"]].copy()
				df.at[0,"Beam Energy"] = self.EGAM_TABLE[ebin] # Text only in first row of table
				df_latexCols.rename(columns={"xsec_tot":"$\dfrac{d\sigma}{dt}$", "statErr_tot":"$\delta_{\text{stat.}}$", "Syst_ptp":"$\delta_{\text{syst.}}$","FluxUncert":"$\delta_{\text{flux}}$"},inplace=True)
				# df_latexCols.rename(columns={"ebin":"EBIN"})
				self.jz_COMBINED_DF_to_latex(df_latexCols,ebin)
		if(saveLowE):
			csv_file_LE = "xsec_results/COMBINED_RESULTS_LE_ebin0to_ebin22.csv"
			if(not os.path.exists(csv_file_LE)):
				print("ERROR: could not find low E combined results!!!!")
				sys.exit()
			df_LE = pd.read_csv(csv_file_LE,index_col="totbin")
			df_LE.replace([-1000.],0.,inplace=True) # For graphing we set these to zero. For tables we want these to be zero.
			df_LE["Syst_ptp"] = df_LE[["systErrLo_tot", "systErrHi_tot"]].max(axis=1)
			with pd.option_context("max_colwidth", 1000):
				df_LE["cos($\theta_{cm}$)"]  = "$"+df_LE["tbinLo"].round(2).astype(str)+"<\cos(\theta)<"+df_LE["tbinHi"].round(2).astype(str)+"$"
				# df_LE["Syst_HiLo"] = "\scriptsize{ ${\text{+"+df_LE[systHiColName].round(3).astype(str)+"} \\atop \text{-"+df_LE[systLoColName].round(3).astype(str)+"}}$ }"
				df_LE["FluxUncert"] = "tbd" # Empy initially
				df_LE["W"] = "" # Empy initially
				# Beam energy column: only fill once per ebin (at FIRST entry)
				for ebin in range(0,self.NUM_W_BINS):
					df_LE.at[ebin*100,"W"] = GetLowETitle(ebin,True) # Text only in first row of table (1, 101, 201, etc --- recall that we skip tbin 0)
				# Now, split by energy bin to create tables that will fit on a single page
				print("NOTE: SKIPPING LOW E bin 0 IN OUTPUT TABLES FOR NOW!!!")
				for ebin in range(1,self.NUM_W_BINS):
					df_latexCols = df_LE[df_LE["ebin"]==ebin].copy()
					df_latexCols = df_latexCols[["W","cos($\theta_{cm}$)","xsec_tot","statErr_tot","Syst_ptp"]].copy()
					df_LE.at[0,"W"] = GetLowETitle(ebin,True) # Text only in first row of table
					df_latexCols.rename(columns={"xsec_tot":"$\dfrac{d\sigma}{d\Omega}$", "statErr_tot":"$\delta_{\text{stat.}}$", "Syst_ptp":"$\delta_{\text{syst.}}$","FluxUncert":"$\delta_{\text{flux}}$"},inplace=True)
					# df_latexCols.rename(columns={"xsec_tot":"$\dfrac{d\sigma}{d\Omega}$", "statErr_tot":"$\delta_{\text{stat.}}$", "Syst_HiLo":"$\delta_{\text{syst.}}$","FluxUncert":"$\delta_{\text{flux}}$"},inplace=True)
					# df_latexCols.rename(columns={"ebin":"EBIN"})
					self.jz_COMBINED_DF_to_latex(df_latexCols,ebin,standardE=False)

	# Assuming we have a dataframe with only the columns we want
	# Tweaks the output a bit too
	# Number of columns is hardcoded!!
	def jz_COMBINED_DF_to_latex(self, df, ebin, standardE=True):

		init_dir = os.getcwd()
		os.chdir(self.TEX_RESULT_DIR)

		texFileName_tmp = "FinalResultsTable_tmp.tex"
		outputFilename = "FinalResultsTable.tex"
		# Create latex table that still needs some formatting
		df.to_latex(texFileName_tmp, index=False, float_format="%.3f",escape=False)  # caption and labels available only in python 3

		caption, label = "", ""
		if (standardE):
			caption = "{\small Differential cross section results for the energy range " + df.at[ebin * 100 + 1, "Beam Energy"] + "~GeV and range of Mandelstam \mt listed (in units of \gevsq). Differential cross sections and corresponding uncertainties are given in units of nb/\gevsq.}"
			label = "tab:final_results_ebin" + str(ebin)
		if (not standardE):
			caption = "{\small Differential cross section results for the W range " + df.at[ebin * 100 + 1, "W"] + "~GeV and in bins of $\cos(\\theta_{cm})$. Differential cross sections and corresponding uncertainties are given in units of nb/steradian}"
			label = "tab:final_results_LE_ebin" + str(ebin)
		# Add some custom formatting to the table
		if (ebin == 0 and standardE):
			f_out = open(outputFilename, "w")  # Overwrite for first ebin
		else:
			f_out = open(outputFilename, "a")  # Append for subsequent ebins
		# Add stuff to front
		f_out.write("\\begin{table}[h]\n")
		f_out.write("	\centering\n")
		f_out.write("	\small\n")
		f_out.write("	\\begin{tabular}{lcccrc}\n")
		# There's a bug in this version of pandas where \midrule shows up in wrong place, handle first few rows by hand
		with open(texFileName_tmp) as f:
			lines = f.readlines()
			f_out.write("\t\t" + lines[1])  # line: \toprule
			f_out.write("\t\t" + lines[2])  # line: [column names]
			f_out.write("\t\t\midrule\n")
			f_out.write("\t\t" + lines[3])  # line: [column names]
			for i, line in enumerate(lines[5:-1]): f_out.write("\t\t" + line)
		f_out.write("\t\end{tabular}\n")
		f_out.write("\t\caption{" + caption + "}\n")
		f_out.write("\t\label{" + label + "}\n")
		f_out.write("\end{table}\n\n")
		os.remove(texFileName_tmp)
		os.chdir(init_dir)


	# Divide df1 by df2 for a given quantity and error
	# # Result added to a new column, both df1 and df2
	def DivideDF1_by_DF2(self,mode_run_tag1,mode_run_tag2,yvar="xsec",yvar_Err="xsec_StatErr"):
		
		df1 = self.df_dict[mode_run_tag1]
		df2 = self.df_dict[mode_run_tag2]
		
		colname     = yvar+"_"+mode_run_tag1+"_over_"+mode_run_tag2
		colnameErr  = yvar+"Err_"+mode_run_tag1+"_over_"+mode_run_tag2
		
		df1[colname]    = df1[yvar]/df2[yvar]
		df1[colnameErr] = df1[colname]* np.sqrt( (df1[yvar_Err]/df1[yvar])**2. + (df2[yvar_Err]/df2[yvar])**2.)
		df2[colname]    = df1[yvar]/df2[yvar]
		df2[colnameErr] = df2[colname]* np.sqrt( (df1[yvar_Err]/df1[yvar])**2. + (df2[yvar_Err]/df2[yvar])**2.)
		
		return colname,colnameErr



# Non-class functions along for the ride:
def jz_DF_to_numpy(df,colname): return np.array(df[colname].to_numpy())

def extract_from_graphkey(key):
	# formed from mode+"_"+run+"_"+tag+"_"+yvar+"_"+str(ebin)
	key_split = key.split("_")
	mode,run,tag= key_split[0],key_split[1],key_split[2]
	return mode,run,tag

LOWE_W_DIVIDER = np.array([(2.52+i*0.04) for i in range(0,23+1)]) # Roughly 2.92-5.8. Although we have data+MC beyond in low E runs, no PS flux.
def GetLowETitle(wbin,latex_ver=False):
	wlo,whi = LOWE_W_DIVIDER[wbin],LOWE_W_DIVIDER[wbin+1]
	if(latex_ver): return "$"+str(round(wlo,2))+" < W < "+str(round(whi,2))+"$"
	return "#eta Diff. Cross Sec. "+str(round(wlo,2))+" < W < "+str(round(whi,2))+" GeV"


