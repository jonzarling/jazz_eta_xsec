#!/usr/bin/env python

#PyRoot file created from template located at:
from optparse import OptionParser
import argparse, os, os.path, sys, time, subprocess, glob, signal, shutil
import numpy as np
import pandas as pd
from numba import vectorize
from math import floor
import gluupy_histmaker as gp
import psutil

from ROOT import TH1F, TH2F, TH3F, TFile, TDirectory, gDirectory
from ROOT import gROOT,gSystem,gDirectory,TDirectoryFile

# DEFINE ALL GLOBALS HERE!
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
M_PROTON, M_ETA, M_PIQ, M_PI0 = 0.938272, 0.547862, 0.13957039, 0.1349768

# Each cross section mode inherits from here
class jzXSecBase(object):

	# DEFAULT INPUT DIRECTORIES
	TOP_DIR               = "/w/halld-scshelf2101/home/jzarling/jzXSecTools/"
	CXX_DIR               = TOP_DIR+"/lib/" # Default, files from other locations can be specified if we want
	DATA_TREES_TOPDIR     = TOP_DIR+"/root_file_links/DATA/" # Default, files from other locations can be specified if we want
	MCRECON_TREES_TOPDIR  = TOP_DIR+"/root_file_links/MC/" # Default, files from other locations can be specified if we want
	FLUX_HISTS_LOC        = TOP_DIR+"/flux/" # Default, files from other locations can be specified if we want
	THEORY_DIR            = TOP_DIR+"/theory/"
	EXPT_DIR              = TOP_DIR+"/pastExpt/"
	# DEFAULT OUTPUT DIRECTORIES
	HIST_STANDARD_DIR     = TOP_DIR+"/hist_files/"
	THROWN_ROOT_HISTS_LOC = TOP_DIR+"/thrown_hists/"
	NOTE_PLOTS_DIR        = TOP_DIR+"/fig_tables/"
	XSEC_RESULT_DIR       = TOP_DIR+"/xsec_results/"
	FITTED_HIST_PLOTDIR   = TOP_DIR+"/more_plots/fit_plots/"
	TEX_RESULT_DIR        = TOP_DIR+"/fig_tables/pdf/"
	# More globals

	def __init__(self,run,tag):

		self.VERBOSE = False

		self.run = run
		self.tag = tag
		self.meson = ""  # Not used for eta. But denotes meson (e.g. eta in the case of g p -> eta p)
		self.mode  = ""  # Decay mode of meson (e.g. gg in the case of exclusive eta production)
		self.binning_type = "ET" # "ET" or "WCT"
		self.ebin_lo = -1 # To be modified later
		self.ebin_hi = -1 # To be modified later
		self.tbin_fit_lo = -1 # To be modified later
		self.tbin_fit_hi = -1 # To be modified later
		self.max_entries=-1
		self.FIT_ALL_VARIATIONS = True if self.tag == "nominal" else False # By default

		# Variables that will be reused across different function calls
		# For histogramming step
		self.all_cut_list = []
		self.NUM_MASS_BINS = -1
		self.N_gammas, self.N_pip, self.N_pim, self.N_pi0 = 0,0,0,0 # Assuming we only have one proton and don't have to specify
		self.runs_to_use, self.runs_to_exclude = [], []
		self.additional_cut_list = []
		self.gen_diagnostic_hists = False
		self.diag_bin_override = -1
		self.diagnostic_hists_sideband = False
		self.skipDataHists, self.skipMCHists = False, False
		self.histfileToFit         = self.HIST_STANDARD_DIR+tag+".root"
		self.root_subdir  = "" # Folder name within ROOT file
		self.additional_hists_fname  = self.HIST_STANDARD_DIR+tag+"_ExtendedHists.root"
		self.root_infile_data = ""
		self.root_infile_mc = ""
		# # For fitting step
		self.df_column_labels = [] # Columns defined below in fit step
		self.save_fit_lvl = 0 # 0=none saved, 1=save default hist, 2=save all fit variations
		self.mcthrown_scalefactor = 1. # Multiply thrown (e.g. use 0.2 to in case of 10 M recon sample, 50 M thrown hists scaled down)
		# For thrown MC generating + retrieving
		self.thrown_hists_mc = ""
		self.THROWN_MC_SEARCH_STRING = ""
		# # The big guy
		self.branches_dict = {} # Dictionary of numpy arrays. Potentially huuuuuge.
		self.branches_to_skip = [""]
		self.branches_to_skip = [""]

	# By default: generate flux and hists files if not found, make histograms, fit histograms, save output
	def DoAnalSteps(self,MakeHists=True,FitHists=True,
					# By default, don't check for flux files AND NOT force them to be remade
	                FluxGenCheck=False,ForceRegenFlux=False,
					# By default, check that
					ThrownGenCheck=True,ForceRemakeThrownHists=False):

		# Check the inputs supplied above, do any additional setup
		self.CheckInputs()
		if self.do_bggen_study: self.skipDataHists=True

		# Run list is needed, except for FitHists step
		self.GetRunList() # Add argument True to print runs and exit
		# Now to make histograms
		if MakeHists:
			self.MakeXSecHists()
		# Need flux and thrown files before fitting, create if needed
		self.flux_fname = self.FLUX_HISTS_LOC+"flux_"+self.meson+"_"+self.run+"_"+str(len(self.runs_to_use))+"runs.root"
		# Check for flux histograms, generate if needed or forcing to
		if FluxGenCheck or ForceRegenFlux or FitHists: self.GenFluxRunSep(self.run, ForceRegenFlux)
		# Check for thrown histograms, generate if needed or forcing to
		if ThrownGenCheck or FitHists or ForceRemakeThrownHists: self.ProcessThrownHists(ForceRemakeThrownHists)
		# Now to fit histograms
		if FitHists:  self.FitXSecHists()

		# Reset just in case we call things more than once before deleting object, might not be all though...
		self.runs_to_use =[]


	def MakeXSecHists(self):

		# NOTE ABOUT CUTS!
		# self.all_cut_list is defined above in GetDefaultCuts
		# Add interface cuts, if desired
		if self.UseFCALCut:
			for i in range(1,self.N_gammas+1): self.all_cut_list.append(["g"+str(i)+"_theta_deg","g"+str(i)+"_theta_deg > ",self.inner_fiducial_cut])
		if self.UseInterfaceCut:
			for i in range(1,self.N_gammas+1): self.all_cut_list.append(["g"+str(i)+"_theta_deg","g"+str(i)+"_theta_deg gt_abs ",10.8,self.interface_fiducial_cut_FullWidth])
		# Add excluded runs to our cut list
		for run in self.runs_to_exclude: self.all_cut_list.append(["run","run != ",run])
		# Non-default cuts are added to cut list here
		for cut in self.additional_cut_list: self.all_cut_list.append(cut)

		# Add "short" to name if debugging and not using all entries
		if self.max_entries!=-1 or (self.ebin_hi - self.ebin_lo + 1)!=self.NUM_E_BINS or (self.tbin_hist_hi - self.tbin_hist_lo + 1)!=self.NUM_T_BINS:
			self.histfileToFit  = self.HIST_STANDARD_DIR+self.tag+"_SHORT.root"
			self.additional_hists_fname = self.HIST_STANDARD_DIR+self.tag+"_ExtendedHists_SHORT.root"

		t_hist_start = time.perf_counter()

		# Some quantities only need calculating if specified for cuts or diagnostic hists
		# (faster if we can skip reading/calculating these)
		calc_theta,get_pi0_invmass,get_Ephotons_sum = False,False,False
		for cut in self.all_cut_list:
			if "theta" in cut[0]: calc_theta=True
			if "pi0" in cut[0] and "3pi" in self.mode:   get_pi0_invmass=True
			if "Ephotons" in cut[0] or "p4_g1_kin__E" in cut[0]:   get_Ephotons_sum=True
		if self.gen_diagnostic_hists:  calc_theta=True

		# DEFINE HISTOGRAMS we'll fill. Most hists will be made twice: once for data and once for MC
		h_nruns = TH1F("h_nruns","Number of runs used",1,0,1)
		h_nruns.SetBinContent(1,len(self.runs_to_use))
		for sample in ["DATA","MC"]:
			isMC = True if sample=="MC" else False

			print("Making general histograms for sample: " + sample)
			if self.skipDataHists and sample=="DATA": continue
			if self.skipMCHists   and sample== "MC":   continue
			# Histograms added for all meson / decay channels
			h_dict = {} # All the histograms that go into the standard output file
			h_dict_diagnostic = {} # All the histograms that go into the diagnostic output file
			h_dict["h_egamma_"+sample]              = TH1F("h_egamma_"+sample,"",self.NUM_MASS_BINS//4,6.,12.)
			h_dict["h_egammaAllCombos_"+sample]     = TH1F("h_egammaAllCombos_"+sample,"",self.NUM_MASS_BINS//4,3.,12.)
			h_dict["h_RFDeltaT_"+sample]            = TH1F("h_RFDeltaT_"+sample,";#Delta t (RF-tag)",1000,-25.,25.)
			h_dict["h_RFDeltaT_accidsub_"+sample]   = TH1F("h_RFDeltaT_accidsub_"+sample,";#Delta t (RF-tag)",1000,-25.,25.)
			h_dict["h_accidweight_"+sample]         = TH1F("h_accidweight_"+sample,";Accidental Weight Factor)",1000,-1.,2.)
			if calc_theta:    h_dict["h_gamma_theta_postcuts" + sample]      = TH1F("h_gamma_theta_postcuts" + sample, "Photon #theta kinfit (degrees)", 300, 0., 30.)
			if calc_theta:    h_dict["h_gamma_theta_meas_postcuts" + sample] = TH1F("h_gamma_theta_meas_postcuts" + sample, "Photon #theta measured (degrees)", 300, 0., 30.)
			if(sample=="MC"): h_dict["h_L1TriggerBits_"+sample]            = TH1F("h_L1TriggerBits_"+sample,"Trigger Bits",100,0,100)
			# Create/modify histograms for different binning style
			if self.binning_type== "WCT":
				del h_dict["h_egamma_"+sample]
				h_dict["h_egamma_"+sample] = TH1F("h_egamma_"+sample,"",self.NUM_MASS_BINS//4,2.5,8.5)
				h_dict["h_W_"+sample] = TH1F("h_W_"+sample,"",self.NUM_MASS_BINS//4,2.5,6.5)

			# Add more meson-specific histograms
			print("Making decay-mode specific hists...")
			self.CreateMesonSpecificHists(h_dict,h_dict_diagnostic,sample) # Including diagnostic hists

			# Define branches to read in (in meson file)
			branch_names_to_use = self.GetBranchNames(calc_theta,get_pi0_invmass,get_Ephotons_sum,self.gen_diagnostic_hists,isMC=isMC)
			for b in self.branches_to_skip:
				if b in branch_names_to_use: branch_names_to_use.remove(b)
			if(isMC and self.UseL1TrigCut): self.all_cut_list.append(["L1TriggerBits","L1TriggerBits > ",0.0001])

			# Finally! Onto actually parsing the ROOT file
			file_to_read = self.root_infile_data if sample=="DATA" else self.root_infile_mc
			print("Opening file: " + file_to_read + "...")
			self.branches_dict = gp.GetBranchesUproot(file_to_read,self.max_entries,branch_names_to_use) # Arguments: filename, OPTIONAL: max entries to parse (default=-1 => all entries), OPTIONAL: list of string branchnames to retrieve (default=get all branches)

			# Add variables required to apply cuts
			t0 = time.perf_counter()
			print("Calculating pre-cut branches...")
			self.CalcAdditionalBranchesPreCuts(calc_theta,get_Ephotons_sum) # Including some meson-agnostic diagnostic vars
			self.PruneBranches("precut")
			print("Time to calculate pre-cut branches: " + str(time.perf_counter()-t0))

			# Fill any pre-cut histograms
			gp.FillHistFromBranchDict(h_dict["h_egammaAllCombos_"+sample],self.branches_dict,"p4_beam__E",DoFSWeighting=False)

			# Apply cuts and calculate FS weights (do earlier on to save VMEM)
			t0 = time.perf_counter()
			self.branches_dict = gp.ApplyCutsReduceArrays(self.branches_dict,self.all_cut_list)
			print("Time to get FS weights: " + str(time.perf_counter()-t0))

			# On-the-fly branches that we'll calculate, incluing diagnostic plot variables
			print("Calculating branches post-cuts...")
			self.CalcAdditionalBranchesPostCuts(get_Ephotons_sum) # Including some meson-agnostic diagnostic vars

			# Fill post-fiductial cut histograms
			gp.FillHistFromBranchDict(h_dict["h_egamma_"+sample],self.branches_dict,"p4_beam__E")
			gp.FillHistFromBranchDict(h_dict["h_RFDeltaT_"+sample],self.branches_dict,"DeltaT_RF",DoAccidentalSub=False,DoFSWeighting=False)
			gp.FillHistFromBranchDict(h_dict["h_RFDeltaT_accidsub_"+sample],self.branches_dict,"DeltaT_RF",DoFSWeighting=False)
			gp.FillHistFromBranchDict(h_dict["h_accidweight_"+sample],self.branches_dict,"accidweight",DoAccidentalSub=False,DoFSWeighting=False)
			if calc_theta:
				for i in range(1,self.N_gammas+1): gp.FillHistFromBranchDict(h_dict["h_gamma_theta_postcuts"+sample],self.branches_dict,"g"+str(i)+"_theta_deg")
				for i in range(1,self.N_gammas+1): gp.FillHistFromBranchDict(h_dict["h_gamma_theta_meas_postcuts"+sample],self.branches_dict,"g"+str(i)+"_theta_deg_meas")
			if(sample=="MC"): gp.FillHistFromBranchDict(h_dict["h_L1TriggerBits_"+sample],self.branches_dict,"L1TriggerBits")

			# Fill meson / decay mode specific histograms
			self.FillMesonSpecificHists(h_dict,h_dict_diagnostic,sample)
			del self.branches_dict

			# Overwrite anything in ROOT folder corresponding to meson/mode/run-period
			if sample== "DATA" or self.skipDataHists: h_dict["h_nruns"]=h_nruns
			cleanDir = True if (sample=="DATA" or self.skipDataHists) else False
			gp.SaveAllHists(self.histfileToFit,hist_list=sorted(list(h_dict.values())),open_opt="UPDATE",subdirname=self.outfile_root_subdir,clearSubDir=cleanDir) #Saves ALL histograms opened/created to this point. Print overall processing rate
			if self.gen_diagnostic_hists: gp.SaveAllHists(self.additional_hists_fname, hist_list=sorted(list(h_dict_diagnostic.values())), open_opt="UPDATE", subdirname=self.outfile_root_subdir, clearSubDir=cleanDir) #Saves ALL histograms opened/created to this point. Print overall processing rate
			del h_dict

	def FitXSecHists(self):

		# Print whether nominal is being used or not
		if self.tag== "nominal": print("TAG \"nominal\" SELECTED! Will fit all histogram variations")

		# Verify all histograms to fit are found
		if not os.path.exists(self.histfileToFit):
			print("ERROR: file containing histograms to fit not found!! Exiting...")
			print("file: " + self.histfileToFit)
			sys.exit()

		# Load pre-compiled fit function
		gROOT.ProcessLine(".L "+self.CXX_DIR+"JZCustomFunctions.cxx+")
		gSystem.Load(self.CXX_DIR+"JZCustomFunctions_cxx.so")
		# print("GOT HERE")

		# Open histogram file, check that all hists can be found
		f = TFile.Open(self.histfileToFit)
		print("Opening histogram file for fitting...")
		self.h2_data_list,self.h2_MC_list=[],[]
		h2_NFCAL_avg_list=[]
		for ebin in range(self.ebin_lo,self.ebin_hi+1):
			# Not sure if can't clone or can't append a None type object.
			# Either way, if histogram doesn't exist, we're going to fail
			topdir = self.mode+"_"+self.run+"/"
			self.h2_data_list.append(  f.Get(topdir+"/h2_eta_kin_ebin"+str(ebin)+"_DATA").Clone())
			self.h2_MC_list.append(    f.Get(topdir+"/h2_eta_kin_ebin"+str(ebin)+"_MC").Clone())
			h2_NFCAL_avg_list.append(  f.Get(topdir+"/h2_NFCAL_ebin"+str(ebin)+"_DATA").Clone())
		for i in range(len(self.h2_data_list)):
			CheckTObjectExists(self.h2_data_list[i],"h2_data")
			CheckTObjectExists(self.h2_MC_list[i],"h2_MC")
			CheckTObjectExists(h2_NFCAL_avg_list[i],"h2_NFCAL_avg")
		nruns_str   = str(int((f.Get(topdir+"h_nruns").GetBinContent(1) + 0.001))) # Fudge factor because we're rounding
		print("Histograms found for " + nruns_str + " runs")

		# ROOT FILE OUTPUT (DF mostly supercedes, but useful for quick checks)
		f_outname = self.XSEC_RESULT_DIR+"xsec_results_"+self.meson+"_"+self.tag+".root"
		f_out_dir = self.run+"_"+self.mode

		# Setup pandas DF
		# One df per mode, run, tag
		# Rows: span t-bins
		# Columns: different fit result quantities
		self.df_column_labels = ["totbin","ebin","tbin","variation","xsec","xsec_StatErr","xsec_SystErrLo_fit","xsec_SystErrHi_fit"]
		self.df_column_labels+= ["avgSigYield","avgSigYield_StatErr","avgSigYield_SystErrLo","avgSigYield_SystErrHi"]
		self.df_column_labels+= ["effic","lumi","N_MCrecon","N_MCthrown","tbinLo","tbinHi"]
		self.df_column_labels+= ["NFCAL_avg","NBCAL_avg"]
		self.df_column_labels+= ["Case0_SigYield","Case1_SigYield","Case2_SigYield",]
		self.df_column_labels+= ["Case0_SigYieldErrBinSum","Case1_SigYieldErrBinSum","Case2_SigYieldErrBinSum",]
		self.df_column_labels+= ["Case0_SigFitErr","Case1_SigFitErr","Case2_SigFitErr",]
		self.df_column_labels+= ["BkgYield_Case0","BkgYield_Case1","BkgYield_Case2",]
		self.df_column_labels+= ["Case0_HistSumErrSigBkg","Case1_HistSumErrSigBkg","Case2_HistSumErrSigBkg",]
		self.df_column_labels+= ["Case0_Chi2NDF","Case1_Chi2NDF","Case2_Chi2NDF",]
		self.df_column_labels+= ["Case2_GausSigmaLarger"]
		self.df_column_labels+= ["YieldAboveSig"]
		df = pd.DataFrame(columns=self.df_column_labels)

		# Now fit!
		startTime = time.perf_counter()
		for i,ebin in enumerate(range(self.ebin_lo,self.ebin_hi+1)):
			for tbin in range(self.tbin_fit_lo,self.tbin_fit_hi+1):
				h_data = self.h2_data_list[i].ProjectionX("h_data_ebin"+str(ebin)+"_tbin"+str(tbin),tbin+1,tbin+1)
				h_MC   = self.h2_MC_list[i].ProjectionX( "h_MC_ebin"  +str(ebin)+"_tbin"+str(tbin),tbin+1,tbin+1)
				df = self.FitXSecHistsOneBin(h_data,h_MC,ebin,tbin,df)

		print("Done with fitting! Calculating xsec and related quantities...")


		# Get thrown and flux files
		f_FLUX      = TFile.Open(self.flux_fname)
		f_thr       = TFile.Open(self.thrown_hists_mc)
		CheckTObjectExists(f_FLUX,"f_FLUX")
		CheckTObjectExists(f_thr,"f_thr")
		LUMI_ARR_FROM_FILE = self.GetFluxArrFromFile(f_FLUX)
		# Calculate thrown numbers on-the-fly, put in dict with same indexing as df
		THROWN_VALS        = {}
		h_thr = f_thr.Get("h_Evst_accepted") if self.binning_type=="ET" else f_thr.Get("h_WvsCT_accepted")
		for i,ebin in enumerate(range(self.ebin_lo,self.ebin_hi+1)):
			for tbin in range(self.tbin_fit_lo,self.tbin_fit_hi+1):
				j    = ebin*100+tbin
				if self.binning_type== "ET":  THROWN_VALS[j] = self.getThrownInETRange(h_thr, self.EBIN_DIVIDER[ebin], self.EBIN_DIVIDER[ebin + 1], self.TBINS_LO[ebin][tbin], self.TBINS_HI[ebin][tbin])
				if self.binning_type== "WCT": THROWN_VALS[j] = self.getThrownInWCTRange(h_thr, self.EBIN_DIVIDER[ebin], self.EBIN_DIVIDER[ebin + 1], self.TBINS_LO[ebin][tbin], self.TBINS_HI[ebin][tbin])

		# Add thrown, effic, flux, NFCAL_avg, NBCAL_avg, and xsec columns to df
		for i in df.index.tolist():
			totbin = df.loc[i]["totbin"]
			ebin,tbin = df.loc[i]["ebin"], df.loc[i]["tbin"]
			df.at[i,"lumi"]=LUMI_ARR_FROM_FILE[ebin]
			df.at[i,"N_MCthrown"]=THROWN_VALS[totbin]*self.mcthrown_scalefactor
			h_NFCAL=h2_NFCAL_avg_list[ebin-self.ebin_lo].ProjectionX("tmp_"+str(tbin),tbin+1,tbin+1)
			df.at[i,"NFCAL_avg"]     = GetNFCALAvgFromTH1F(h_NFCAL)
			df.at[i,"YieldAboveSig"] = jzHistogramSumIntegral(h_NFCAL)
		df.loc[df["NFCAL_avg"] < 0., "NFCAL_avg" ] = 0. # Very rare, but sometimes accidental subtraction can lead to negative values.
		# Get effic (avoiding divide by 0 error)
		df.loc[df["N_MCthrown"] < 1., "N_MCthrown" ] = -1000. # Avoid dividing by zero: set negative instead
		df["effic"] = df["N_MCrecon"]/df["N_MCthrown"]
		df["NBCAL_avg"] = self.N_gammas- df["NFCAL_avg"]


		# print "GETTING XSEC..."
		df["xsec"]               = self.GetDiffXSec(df["avgSigYield"],   df["effic"],df["lumi"], (df["tbinHi"]-df["tbinLo"]) )
		df["xsec_StatErr"]       = self.GetDiffXSec(df["avgSigYield_StatErr"],df["effic"],df["lumi"], (df["tbinHi"]-df["tbinLo"]) )
		df["xsec_SystErrHi_fit"] = self.GetDiffXSec(df["avgSigYield_SystErrHi"],df["effic"],df["lumi"], (df["tbinHi"]-df["tbinLo"]) )
		df["xsec_SystErrLo_fit"] = self.GetDiffXSec(df["avgSigYield_SystErrLo"],df["effic"],df["lumi"], (df["tbinHi"]-df["tbinLo"]) )
		df.replace([np.nan, np.inf, -np.inf], 0., inplace=True) # Remove nan and infs

		# If fitting multiple variations, first save a copy with all fits info, then drop
		if self.FIT_ALL_VARIATIONS:
			print("Saving all fits to csv...")
			df.to_csv(self.XSEC_RESULT_DIR+"/ALL_FITS/"+self.mode+"_"+self.run+"_"+self.tag+"_RESULTS_ebin"+str(self.ebin_lo)+"to_ebin"+str(self.ebin_hi)+".csv")
			df = df[ (df["variation"]=="nominal") ].reset_index(drop=True).copy()

		# Save nominal fit results
		df.set_index("totbin",inplace=True)
		df.to_csv(self.XSEC_RESULT_DIR+self.mode+"_"+self.run+"_"+self.tag+"_RESULTS_ebin"+str(self.ebin_lo)+"to_ebin"+str(self.ebin_hi)+".csv")

		print("Done fitting histograms")
		print("Time to fit "+str(len(df))+" histograms: " + str(time.perf_counter()-startTime))

	def GetRunList(self,print_runs_to_use=False):

		# First get all runs from rcdb
		import rcdb
		RCDB_QUERY = ""
		RUN_LO,RUN_HI=0,0
		if self.run== "SP17":
			RCDB_QUERY = "@is_production and @status_approved"
			RUN_LO,RUN_HI=30274,31057
		elif self.run== "SP18":
			RCDB_QUERY = "@is_2018production and @status_approved"
			RUN_LO,RUN_HI=40856,42577
		elif self.run== "FA18":
			RCDB_QUERY = "@is_2018production and @status_approved and beam_on_current > 49"
			RUN_LO,RUN_HI=50677,52715
		elif self.run== "FA18LE":
			RCDB_QUERY = "@is_2018production and @status_approved and beam_on_current < 49"
			RUN_LO,RUN_HI=51384,51457
		elif self.run== "SP20":
			RCDB_QUERY = "@is_dirc_production and @status_approved"
			RUN_LO,RUN_HI=71350,73266
		else:
			print("ERROR! Invalid run supplied: " + run)
			raise RuntimeError
		rcdb_conn = rcdb.RCDBProvider("mysql://rcdb@hallddb.jlab.org/rcdb")
		runsFromRCDB = rcdb_conn.select_runs(RCDB_QUERY, RUN_LO, RUN_HI)
		rcdb_run_list = np.array( [ int(str(run)[13:18]) for run in runsFromRCDB] )

		# Then get all runs found in data/MC samples
		# # Check runs in data and MC against one another
		branchname = ["run"]
		data_br = gp.GetBranchesUproot(self.root_infile_data,self.max_entries,branchname,skipStandardBranches=True)
		mc_br   = gp.GetBranchesUproot(self.root_infile_mc,self.max_entries,branchname,skipStandardBranches=True)
		all_data_runs = np.unique(data_br["run"])
		all_mc_runs   = np.unique(  mc_br["run"])
		mask = np.isin(all_data_runs, all_mc_runs, invert=True)

		dataruns_NotIn_rcdb = all_data_runs[np.isin(all_data_runs, rcdb_run_list, invert=True)]
		MCruns_NotIn_rcdb   = all_mc_runs[np.isin(all_mc_runs, rcdb_run_list, invert=True)]
		# Not entirely clear why some approved runs don't appear in data or MC. For data, maybe runs were marked bad at analysis launch but good later? For MC: maybe missing random trigger files?
		rcdbruns_NotIn_data = rcdb_run_list[np.isin(rcdb_run_list, all_data_runs, invert=True)]
		rcdbruns_NotIn_MC   = rcdb_run_list[np.isin(rcdb_run_list, all_mc_runs, invert=True)]

		self.runs_to_exclude = np.unique( np.concatenate( (dataruns_NotIn_rcdb,MCruns_NotIn_rcdb,rcdbruns_NotIn_data,rcdbruns_NotIn_MC) ) )
		# Determine list of accepted runs
		for run in rcdb_run_list:
			if run in self.runs_to_exclude: continue
			else: self.runs_to_use.append(run)
		print("Excluding runs: " + str(self.runs_to_exclude))

		print("Number of data runs: " + str(len(all_data_runs)))
		print("Number of MC runs: "   + str(len(all_mc_runs)))
		print("Number of rcdb runs: " + str(len(rcdb_run_list)))
		print("ALL DATA RUNS NOT IN RCDB: " + str(dataruns_NotIn_rcdb))
		print("ALL  MC  RUNS NOT IN RCDB: " + str(MCruns_NotIn_rcdb))
		print("ALL  RCDB  RUNS NOT IN DATA: " + str(rcdbruns_NotIn_data))
		print("ALL  RCDB  RUNS NOT IN MC: " + str(rcdbruns_NotIn_MC))
		print("\n\n")

		if print_runs_to_use:
			all_runs_str = ""
			for run in self.runs_to_use: all_runs_str=all_runs_str+" "+str(run)
			print(all_runs_str)
			sys.exit()


	def GenFluxRunSep(self,run,ForceRegenFlux):
		print("Doing flux step....")
		# HADD FLUX FILES (if needed)
		if not os.path.exists(self.flux_fname):
			print("Generating flux hadd file "+self.flux_fname+" ...")
			init_dir = os.getcwd()
			os.chdir(self.FLUX_HISTS_LOC)
			hadd_command = "hadd -f " + self.flux_fname
			run_digit = ""
			if self.run== "SP17":     run_digit= "3"
			elif self.run== "SP18":   run_digit= "4"
			elif "FA18" in self.run: run_digit= "5"
			elif self.run== "SP20":   run_digit= "7"
			all_flux_files = sorted(glob.glob(self.run+"/flux_"+self.meson+"_"+run_digit+"????.root"))
			for run in self.runs_to_use:
				fname = self.run+"/flux_"+self.meson+"_"+str(run)+".root"
				# Add existing file to hadd command, if already exists (and not overwriting)
				if os.path.exists(fname) and not ForceRegenFlux:
					hadd_command+=" "+fname
					print("Flux file found for run " + str(run))
				else:
					self.shell_exec(self.GetFluxShellExecString(run))
					shutil.move(self.run+"/flux_"+str(run)+".root",fname)
					if not os.path.exists(fname):
						print("ERROR: missing thrown histogram file: " + str(fname))
						sys.exit()
			self.shell_exec(hadd_command)
			os.chdir(init_dir)

	def GetFluxArrFromFile(self,f_FLUX):
		N_lumi_bins = self.NUM_E_BINS
		lumi_arr_np = np.zeros(N_lumi_bins)
		h=f_FLUX.Get("tagged_lumi")
		CheckTObjectExists(h,"tagged lumi")

		for ebin in range(0,N_lumi_bins):
			hist_left_edge,hist_right_edge=0.,0.
			if self.binning_type== "ET":  hist_left_edge,hist_right_edge = self.EBIN_DIVIDER[ebin],self.EBIN_DIVIDER[ebin + 1]
			if self.binning_type== "WCT": hist_left_edge,hist_right_edge = calc_E_from_W(self.EBIN_DIVIDER[ebin]), calc_E_from_W(self.EBIN_DIVIDER[ebin + 1])
			lumi_arr_np[ebin]=jzHistogramSumIntegral(h,hist_left_edge,hist_right_edge)
		return lumi_arr_np


	# Meson & decay-mode specific placeholders that MUST be reimplemented in (e.g.) jzEtaMesonXSec
	def ModeChannelInit(self):
		print("ModeChannelInit function expected to be reimplemented, was not found!!!")
		sys.exit()
	def GetDefaultCuts(self):
		print("GetDefaultCuts function expected to be reimplemented, was not found!!!")
		sys.exit()
	def GetETBinWindows(self):
		print("GetETBinWindows function expected to be reimplemented, was not found!!!")
		sys.exit()
	def CreateMesonSpecificHists(self):
		print("CreateMesonSpecificHists function expected to be reimplemented, was not found!!!")
		sys.exit()
	def FillMesonSpecificHists(self):
		print("FillMesonSpecificHists function expected to be reimplemented, was not found!!!")
		sys.exit()
	def CalcAdditionalBranchesPreCuts(self):
		print("CalcAdditionalBranchesPreCuts function expected to be reimplemented, was not found!!!")
		sys.exit()
	def CalcAdditionalBranchesPostCuts(self):
		print("CalcAdditionalBranchesPostCuts function expected to be reimplemented, was not found!!!")
		sys.exit()
	def GetBranchNames(self):
		print("GetBranchNames function expected to be reimplemented, was not found!!!")
		sys.exit()
	def PruneBranches(self):
		print("PruneBranches function expected to be reimplemented, was not found!!!")
		sys.exit()
	def Get_ebin(self): # Actually, a hinky global function will eventually get used, not a class one. Just leaving here for code documentation.
		print("Get_ebin function expected to be reimplemented, was not found!!!")
		sys.exit()
	def Get_tbin(self): # Actually, a hinky global function will eventually get used, not a class one. Just leaving here for code documentation.
		print("Get_tbin function expected to be reimplemented, was not found!!!")
		sys.exit()
	def GenTBinArrays(self):
		print("GenTBinArrays function expected to be reimplemented, was not found!!!")
		sys.exit()
	def GetFitRangesByMode(self):
		print("GetFitRangesByMode function expected to be reimplemented, was not found!!!")
		sys.exit()
	def GetFluxShellExecString(self):
		print("GetFluxShellExecString function expected to be reimplemented, was not found!!!")
		sys.exit()
	def ProcessThrownHists(self):
		print("ProcessThrownHists function expected to be reimplemented, was not found!!!")
		sys.exit()
	def getThrownInETRange(self):
		print("getThrownInETRange function expected to be reimplemented, was not found!!!")
		sys.exit()
	def getThrownInWCTRange(self):
		print("getThrownInWCTRange function expected to be reimplemented, was not found!!!")
		sys.exit()


	# HELPER FUNCTIONS BELOW
	# HELPER FUNCTIONS BELOW
	# HELPER FUNCTIONS BELOW
	def CheckInputs(self):
		if self.binning_type!= "ET" and self.binning_type!= "WCT":
			print("ERROR: unexpected binning type!\n binning_type supplied: "+self.binning_type)
			sys.exit()

	def shell_exec(self,command, stdin_str=""): #stdin_str for things fed via command line with "<"
		print("COMMAND: " + command)
		command_split_list = command.split() # Vanilla split function filters for arbitrary number of spaces
		if '' in command_split_list: command_split_list.remove('') #Remove empty entries from command, shell doesn't know what to do with these...
		if '\n' in command_split_list: command_split_list.remove('\n') #Remove newline entries from command, shell doesn't know what to do with these...
		status = subprocess.call(command_split_list)
		return

	# Given in nb (but assumes lumi supplied in pb^-1)
	def GetDiffXSec(self,data_yield,effic,lumi,tbin_width):

		# conversion_factor = 1000. if not microbarns else 1.e6 # nanobarns by default
		# print "CONVERSION FACTOR: " + str(conversion_factor)

		conversion_factor = 0.
		if self.binning_type== "ET":    conversion_factor = 1000. # Convert lumi picobarns to nanobarns
		elif self.binning_type== "WCT": conversion_factor = 2 * 3.14159 * 1000. # Want steradian units, lumi picobarns to nanobarns
		else:
			print("ERROR! Incorrect binning type supplied: " + self.binning_type)
			return -1 # Shouldn't happen

		return data_yield / (conversion_factor*effic*lumi*self.BF[self.mode]*tbin_width) # Factor of 1000 to go from pb to nb




def PrintMemUsage(stepNameString):
	print("Step: " + stepNameString)
	print("RSS memory usage in GB: " + str( psutil.Process(os.getpid()).memory_info()[0] / 1024.** 3 ))
	print("VMS memory usage in GB: " + str( psutil.Process(os.getpid()).memory_info()[1] / 1024.** 3 ))
	print("Shared memory usage in GB: " + str( psutil.Process(os.getpid()).memory_info()[2] / 1024.** 3 ))
	print("")

def jzHistogramSumIntegral(h,min_hist_val=-1e9,max_hist_val=1e9):
	sum = 0.0
	for i in range(0,h.GetNbinsX()+2):
		if h.GetBinLowEdge(i)+1e-4>=min_hist_val and (h.GetBinLowEdge(i) + h.GetBinWidth(1))<=max_hist_val+1e-4: sum+=h.GetBinContent(i) #Recall: 0 is underflow, nbinsx is maximum bin, nbinsx+1 is overflow bin. I think.
	return sum

def calc_W_from_E(E):
	return ( (E+0.938272)**2 -E**2 )**0.5
def calc_E_from_W(W):
    return (W**2-0.938272**2)/(2*0.938272)

def CheckTObjectExists(TObj,name):
	if TObj==None:
		print("ERROR object "+str(name)+" does not exist!")
		raise RuntimeError

def GetNFCALAvgFromTH1F(h):
	sum=0.
	if h.Integral()<1.: return 0.
	for ngam in range(0,7): sum+=ngam*h.GetBinContent(ngam+1)
	return sum/h.Integral()


