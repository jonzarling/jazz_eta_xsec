#!/usr/bin/env python3

from jzXSecCoreLib import * # jzXSecBase and globals
from FitXSecHists_2GausSig_ExpBkg import jzHistFitter as jz2GausFit # Fitting function

class jzEtaMesonXSec(jzXSecBase):

	TARGET_Z_MIN, TARGET_Z_MAX = 52.,78.
	RESTversByRunPer ={"SP17":3,"SP18":2,"FA18":2,"FA18LE":2,"SP20":1}
	
	BF = {
		"gg" : 0.3941,
		"3piq" : 0.2292,#*0.9882,      # MC allows pi0 to decay to gamma gamma or to gamma e+e-. BF does not need to be included.
		"3pi0" : 0.3268,#*(0.9882**3), # MC allows pi0 to decay to gamma gamma or to gamma e+e-. BF does not need to be included.
	}
	
	def __init__(self,mode,run,tag):
		super(jzEtaMesonXSec,self).__init__(run,tag) # Arguments that jzXSecCoreLib init needs
		self.meson = "eta" # Denotes decaying meson
		self.mode  = mode # Denotes decaying meson
		self.RESTver = self.RESTversByRunPer[run]
		# Kinematic binning (modified later)
		NUM_E_BINS    = -1
		NUM_T_BINS    = -1
		
		# Variables that will be reused across different function calls
		# HISTOGRAMMING STEP
		self.NUM_MASS_BINS = 1000
		self.UseFCALCut, self.UseInterfaceCut, self.UseL1TrigCut = True, False, True
		self.inner_fiducial_cut, self.interface_fiducial_cut_FullWidth   = 2., 0.8
		self.make_example_plots = False # Make plots for analysis note with ebin3, tbins=4,22,35
		self.outfile_root_subdir = self.mode+"_"+self.run
		
		# FITTING STEP
		self.AddPlottingStuff() # Too long to list here, so it goes at the end
		self.rebin_factor  = 2 
		self.max_nfits     = 2
		self.UseLikFit     = True
		
		# SPECIAL STUDIES AND THE LIKE
		# # Special studies that piggyback on histogramming step
		self.do_cut_flow_study  = False # Add more branches to read in and make more histograms
		self.cut_flow_study_ebinLo, self.cut_flow_study_ebinHi = 8,9
		self.cut_flow_study_tbinLo, self.cut_flow_study_tbinHi = 0,4
		self.do_bggen_study     = False # Add more branches to read in and make more histograms 

		# Final initializing
		self.ModeChannelInit()  # setup more paths that need meson+mode specified
		self.GetETBinWindows()  

		self.GetDefaultCuts()
		
	# Can't define these in __init__ due to inheritance structure, but eh close enough
	def ModeChannelInit(self):
		# IMPORTANT NOTE!!! I haven't used self.N_pip and similar for anything yet... lol
		# Number of particles (used in some for loops)
		if(self.mode == "gg"):   self.N_gammas, self.N_pip, self.N_pim, self.N_pi0 = 2,0,0,0 # Assuming we only have one proton and don't have to specify
		if(self.mode == "3piq"): self.N_gammas, self.N_pip, self.N_pim, self.N_pi0 = 2,1,1,1 # Assuming we only have one proton and don't have to specify
		if(self.mode == "3pi0"): self.N_gammas, self.N_pip, self.N_pim, self.N_pi0 = 6,0,0,3 # Assuming we only have one proton and don't have to specify
		# Hardcoded file locations
		self.root_infile_data     = self.DATA_TREES_TOPDIR+"/eta_"+self.mode+"_"+self.run+"_DATA.root"
		self.root_infile_mc       = self.MCRECON_TREES_TOPDIR+"/eta_"+self.mode+"_"+self.run+"_MCrecon.root"
		self.THROWN_MC_SEARCH_STRING = "/cache/halld/gluex_simulations/REQUESTED_MC/july23_eta_"+self.mode+"_"+self.run+"_????/root/thrown/*.root"
		self.root_subdir  = self.mode+"_"+self.run # Folder name within ROOT file
		if(self.run=="FA18LE"): self.binning_type = "WCT"

		self.run_digit = ""
		if(self.run=="SP17"):     self.run_digit="3"
		elif(self.run=="SP18"):   self.run_digit="4"
		elif("FA18" in self.run): self.run_digit="5"
		elif(self.run=="SP20"):   
			self.run_digit="7"
			self.THROWN_MC_SEARCH_STRING = "/cache/halld/gluex_simulations/REQUESTED_MC/july23_eta_"+self.mode+"_FA19_????/root/thrown/*.root"
		




	def GetDefaultCuts(self):
		self.all_cut_list.extend([
							["x4_prot_meas_z","Proton Z vertex (measured) > ",52.],
							["x4_prot_meas_z","Proton Z vertex (measured) < ",78.],
							["p4_prot_pmag"  ,"Proton momentum > ",0.35],
							["chi2_ndf","chi^2/ndf < ",10.0],
							["ebin","ebin >= ",-0.1],
							["tbin","tbin >= ",-0.1],
						])

		return 
		
	def GetBranchNames(self,calc_theta,get_pi0_invmass,get_Ephotons_sum,get_diagnostic,isMC=False):
		
		all_cut_vars = [cut_list[0] for cut_list in self.all_cut_list]
		
		branch_names_to_use = ["run","event","chi2_ndf", "accidweight","p4_beam__E","x4_beam_t","x4_prot_meas_z","p4_prot_meas_px","p4_prot_meas_py","p4_prot_meas_pz"]
		branch_names_to_use.extend(["minus_t","chi2_ndf_DiffFromMin","DeltaT_RF"])
		branch_names_to_use.extend(["eta_mass_meas","eta_mass_kin"])
		if(self.mode=="3piq" and (get_pi0_invmass or get_diagnostic) ):       branch_names_to_use.extend(["pi0_mass_kin","pi0_mass_meas","FCAL_ShowQuality_g1","FCAL_ShowQuality_g2"])
		if(self.mode=="3pi0" and (get_pi0_invmass or get_diagnostic) ):       branch_names_to_use.extend(["pi0_1_mass_meas","pi0_2_mass_meas","pi0_3_mass_meas","pi0_1_mass_kin","pi0_2_mass_kin","pi0_3_mass_kin"])		
		if(self.binning_type=="WCT"): branch_names_to_use.append("cos_theta_cm")
		if(isMC): branch_names_to_use.append("L1TriggerBits")

		# More misc. non-default branches
		if("FCAL_ShowQuality_g1" in all_cut_vars): branch_names_to_use.extend(["FCAL_ShowQuality_g1","FCAL_ShowQuality_g2"])
		if("g1_FCAL_DOCA" in all_cut_vars and self.mode=="3piq"): branch_names_to_use.extend(["g1_FCAL_DOCA","g2_FCAL_DOCA"])
		if(calc_theta or get_Ephotons_sum):
			for i in range(1,self.N_gammas+1): branch_names_to_use.extend(["p4_g"+str(i)+"_meas_px","p4_g"+str(i)+"_meas_py","p4_g"+str(i)+"_meas_pz","p4_g"+str(i)+"_meas__E"])	
			for i in range(1,self.N_gammas+1): branch_names_to_use.extend(["p4_g"+str(i)+"_kin_px","p4_g"+str(i)+"_kin_py","p4_g"+str(i)+"_kin_pz","p4_g"+str(i)+"_kin__E"])	
		if(get_diagnostic):
			branch_names_to_use.extend(["p4_prot_kin_px","p4_prot_kin_py","p4_prot_kin_pz","p4_prot_kin__E","p4_prot_meas__E"])
			branch_names_to_use.extend(["x4_prot_meas_x","x4_prot_meas_y",])
			if(self.binning_type=="ET"):
				for i in range(1,self.N_gammas+1): branch_names_to_use.extend(["x4_g"+str(i)+"_shower_x","x4_g"+str(i)+"_shower_y","x4_g"+str(i)+"_shower_z"])
			if(self.mode=="3piq"): 
				branch_names_to_use.extend(["p4_pip_kin_px","p4_pip_kin_py","p4_pip_kin_pz","p4_pip_kin__E"])
				branch_names_to_use.extend(["p4_pim_kin_px","p4_pim_kin_py","p4_pim_kin_pz","p4_pim_kin__E"])
				branch_names_to_use.extend(["p4_pip_meas_px","p4_pip_meas_py","p4_pip_meas_pz","p4_pip_meas__E"])
				branch_names_to_use.extend(["p4_pim_meas_px","p4_pim_meas_py","p4_pim_meas_pz","p4_pim_meas__E"])
		if(self.do_bggen_study):    branch_names_to_use.extend(["MC_TopologyType"])
		if(self.do_cut_flow_study): # Cut flow: PID Delta t, dE/dx, MM2, proton_pmag, chi2_ndf, beam energy, vertex z, fiducial_photon_theta
			branch_names_to_use.extend(["p4_prot_meas__E"])
			branch_names_to_use.extend(["p4_prot_meas__E","proton_dEdx_CDC","proton_PIDT_BCAL","proton_PIDT_FCAL","proton_PIDT_TOF","proton_PIDT_SC"])
			for i in range(1,self.N_gammas+1): branch_names_to_use.extend(["g"+str(i)+"_PIDT_BCAL","g"+str(i)+"_PIDT_FCAL",])
			if(self.mode=="3piq"): branch_names_to_use.extend(["pip_PIDT_BCAL","pip_PIDT_FCAL","pip_PIDT_TOF","pip_PIDT_SC"])
			if(self.mode=="3piq"): branch_names_to_use.extend(["pim_PIDT_BCAL","pim_PIDT_FCAL","pim_PIDT_TOF","pim_PIDT_SC"])
			# if(self.mode=="3piq"): branch_names_to_use.extend(["pip_dEdx_CDC","pip_PIDT_BCAL","pip_PIDT_FCAL","pip_PIDT_TOF","pip_PIDT_SC"])
			# if(self.mode=="3piq"): branch_names_to_use.extend(["pim_dEdx_CDC","pim_PIDT_BCAL","pim_PIDT_FCAL","pim_PIDT_TOF","pim_PIDT_SC"])
			if(self.mode=="3piq"): branch_names_to_use.extend(["p4_pip_meas__E","p4_pim_meas__E"])
			if(self.mode=="3piq"): branch_names_to_use.extend(["g1_FCAL_DOCA","g2_FCAL_DOCA",])
			if(self.mode=="3piq" and isMC): branch_names_to_use.extend(["pip_PIDT_TOF_meas","pim_PIDT_TOF_meas",])
		return branch_names_to_use
		
	# ebin, tbin, p4_prot_pmag used for cuts
	def CalcAdditionalBranchesPreCuts(self,calc_theta,get_Ephotons_sum):
		
		all_cut_vars = [cut_list[0] for cut_list in self.all_cut_list]
		all_branches = self.branches_dict.keys()

		self.branches_dict["p4_prot_pmag"]   = np.sqrt( self.branches_dict["p4_prot_meas_px"]**2 + self.branches_dict["p4_prot_meas_py"]**2 + self.branches_dict["p4_prot_meas_pz"]**2)
		if("p4_pip_kin_p" in all_cut_vars): self.branches_dict["p4_pip_kin_p"]  = np.sqrt( self.branches_dict["p4_pip_kin_px"]**2 + self.branches_dict["p4_pip_kin_py"]**2 + self.branches_dict["p4_pip_kin_pz"]**2)
		if("p4_pim_kin_p" in all_cut_vars): self.branches_dict["p4_pim_kin_p"]  = np.sqrt( self.branches_dict["p4_pim_kin_px"]**2 + self.branches_dict["p4_pim_kin_py"]**2 + self.branches_dict["p4_pim_kin_pz"]**2)
		
		if(self.binning_type=="ET"):
			self.branches_dict["ebin"]  = Get_ebin(self.branches_dict["p4_beam__E"])
			self.branches_dict["tbin"]  = Get_tbin(self.branches_dict["ebin"],self.branches_dict["minus_t"])
		if(self.binning_type=="WCT"):
			self.branches_dict["W"]     = np.sqrt((self.branches_dict["p4_beam__E"]+M_PROTON)**2 - self.branches_dict["p4_beam__E"]**2)
			self.branches_dict["ebin"]  = Get_ebin_LE(self.branches_dict["W"])
			self.branches_dict["tbin"]  = Get_tbin_LE(self.branches_dict["ebin"],self.branches_dict["cos_theta_cm"])

		#Calculate theta on the fly, if needed
		if(calc_theta):
			for i in range(1,self.N_gammas+1):
				self.branches_dict["g"+str(i)+"_theta_deg_meas"] = np.arccos( self.branches_dict["p4_g"+str(i)+"_meas_pz"]/self.branches_dict["p4_g"+str(i)+"_meas__E"] )*(180/3.14159)
				self.branches_dict["g"+str(i)+"_phi_deg_meas"]   = np.arctan2( self.branches_dict["p4_g"+str(i)+"_meas_py"],self.branches_dict["p4_g"+str(i)+"_meas_px"] )*(180/3.14159)
				self.branches_dict["g"+str(i)+"_theta_deg"]      = np.arccos( self.branches_dict["p4_g"+str(i)+"_kin_pz"]/self.branches_dict["p4_g"+str(i)+"_kin__E"] )*(180/3.14159)
				self.branches_dict["g"+str(i)+"_phi_deg"]        = np.arctan2( self.branches_dict["p4_g"+str(i)+"_kin_py"],self.branches_dict["p4_g"+str(i)+"_kin_px"] )*(180/3.14159)
				self.branches_dict["g"+str(i)+"_isFCAL"]         = np.where(  self.branches_dict["g"+str(i)+"_theta_deg"]<10.8,1,0)
			# Number of showers in FCAL (skipping unused in combo ones)
			self.branches_dict["NumFCALShowers"]                 = self.branches_dict["g1_isFCAL"]
			for i in range(2,self.N_gammas+1): self.branches_dict["NumFCALShowers"]  += self.branches_dict["g"+str(i)+"_isFCAL"] # Already added g1 above
		if(get_Ephotons_sum):
			self.branches_dict["Ephotons"]   = self.branches_dict["p4_g1_kin__E"]
			for i in range(2,self.N_gammas+1): self.branches_dict["Ephotons"]    += self.branches_dict["p4_g"+str(i)+"_kin__E"]
		for cut in self.all_cut_list:
			if("pip_theta_deg" in cut[0]):
				self.branches_dict["p4_pip_kin_p"]  = np.sqrt( self.branches_dict["p4_pip_kin_px"]**2 + self.branches_dict["p4_pip_kin_py"]**2 + self.branches_dict["p4_pip_kin_pz"]**2)
				self.branches_dict["p4_pim_kin_p"]  = np.sqrt( self.branches_dict["p4_pim_kin_px"]**2 + self.branches_dict["p4_pim_kin_py"]**2 + self.branches_dict["p4_pim_kin_pz"]**2)
				self.branches_dict["pip_theta_deg"] = np.arccos( self.branches_dict["p4_pip_kin_pz"]/self.branches_dict["p4_pip_kin_p"])*(180/3.14159)
				self.branches_dict["pim_theta_deg"] = np.arccos( self.branches_dict["p4_pim_kin_pz"]/self.branches_dict["p4_pim_kin_p"])*(180/3.14159)
			# If PID timing info comes from other subsystem (or no PID info), this will ensure they are NOT cut
			if "_PIDT_" in cut: self.branches_dict[cut][self.branches_dict[cut] <-999.] += 1000.

		
		if(self.do_cut_flow_study or ("DeltaPhi" in all_cut_vars) or ("recM_prot" in all_cut_vars) or ("missing__E" in all_cut_vars) ):
			# Add variables that aren't already calculated
			# Get Missing Mass^2
			if(self.mode=="gg"): 
				self.branches_dict["eta_can_px"]     = self.branches_dict["p4_g1_meas_px"]+self.branches_dict["p4_g2_meas_px"]
				self.branches_dict["eta_can_py"]     = self.branches_dict["p4_g1_meas_py"]+self.branches_dict["p4_g2_meas_py"]
				self.branches_dict["eta_can_pz"]     = self.branches_dict["p4_g1_meas_pz"]+self.branches_dict["p4_g2_meas_pz"]
				self.branches_dict["eta_can__E"]     = self.branches_dict["p4_g1_meas__E"]+self.branches_dict["p4_g2_meas__E"]
			if(self.mode=="3piq"): 
				self.branches_dict["eta_can_px"]     = self.branches_dict["p4_g1_meas_px"]+self.branches_dict["p4_g2_meas_px"]+self.branches_dict["p4_pip_meas_px"]+self.branches_dict["p4_pim_meas_px"]
				self.branches_dict["eta_can_py"]     = self.branches_dict["p4_g1_meas_py"]+self.branches_dict["p4_g2_meas_py"]+self.branches_dict["p4_pip_meas_py"]+self.branches_dict["p4_pim_meas_py"]
				self.branches_dict["eta_can_pz"]     = self.branches_dict["p4_g1_meas_pz"]+self.branches_dict["p4_g2_meas_pz"]+self.branches_dict["p4_pip_meas_pz"]+self.branches_dict["p4_pim_meas_pz"]
				self.branches_dict["eta_can__E"]     = self.branches_dict["p4_g1_meas__E"]+self.branches_dict["p4_g2_meas__E"]+self.branches_dict["p4_pip_meas__E"]+self.branches_dict["p4_pim_meas__E"]
			if(self.mode=="3pi0"): 
				self.branches_dict["eta_can_px"]     = self.branches_dict["p4_g1_meas_px"]+self.branches_dict["p4_g2_meas_px"]+self.branches_dict["p4_g3_meas_px"]+self.branches_dict["p4_g4_meas_px"]+self.branches_dict["p4_g5_meas_px"]+self.branches_dict["p4_g6_meas_px"]
				self.branches_dict["eta_can_py"]     = self.branches_dict["p4_g1_meas_py"]+self.branches_dict["p4_g2_meas_py"]+self.branches_dict["p4_g3_meas_py"]+self.branches_dict["p4_g4_meas_py"]+self.branches_dict["p4_g5_meas_py"]+self.branches_dict["p4_g6_meas_py"]
				self.branches_dict["eta_can_pz"]     = self.branches_dict["p4_g1_meas_pz"]+self.branches_dict["p4_g2_meas_pz"]+self.branches_dict["p4_g3_meas_pz"]+self.branches_dict["p4_g4_meas_pz"]+self.branches_dict["p4_g5_meas_pz"]+self.branches_dict["p4_g6_meas_pz"]
				self.branches_dict["eta_can__E"]     = self.branches_dict["p4_g1_meas__E"]+self.branches_dict["p4_g2_meas__E"]+self.branches_dict["p4_g3_meas__E"]+self.branches_dict["p4_g4_meas__E"]+self.branches_dict["p4_g5_meas__E"]+self.branches_dict["p4_g6_meas__E"]
			self.branches_dict["missing_px"]    = self.branches_dict["p4_prot_meas_px"]+self.branches_dict["eta_can_px"]
			self.branches_dict["missing_py"]    = self.branches_dict["p4_prot_meas_py"]+self.branches_dict["eta_can_py"]
			self.branches_dict["missing_pz"]    = self.branches_dict["p4_beam__E"]-self.branches_dict["p4_prot_meas_pz"]-self.branches_dict["eta_can_pz"]
			self.branches_dict["missing__E"]    = self.branches_dict["p4_beam__E"]+M_PROTON-self.branches_dict["p4_prot_meas__E"]-self.branches_dict["eta_can__E"]
			self.branches_dict["missing_M2"]    = self.branches_dict["missing__E"]**2-self.branches_dict["missing_px"]**2-self.branches_dict["missing_py"]**2-self.branches_dict["missing_pz"]**2
			self.branches_dict["DeltaPhi"]      = np.abs( np.arctan2( self.branches_dict["p4_prot_meas_py"],self.branches_dict["p4_prot_meas_px"] )*(180/3.14159)-np.arctan2( self.branches_dict["eta_can_py"],self.branches_dict["eta_can_px"])*(180/3.14159) )
			self.branches_dict["recM_prot"]     = np.sqrt( (self.branches_dict["p4_beam__E"]+M_PROTON-self.branches_dict["p4_prot_meas__E"])**2-self.branches_dict["p4_prot_meas_px"]**2-self.branches_dict["p4_prot_meas_py"]**2-(self.branches_dict["p4_beam__E"]-self.branches_dict["p4_prot_meas_pz"])**2)
			# Get logified dEdx (also using p)
			# See https://halldweb.jlab.org/wiki/index.php/Spring_2017_Analysis_Launch_Cuts#Track_Energy_Loss_Cuts
			if(self.do_cut_flow_study):
				if("proton_dEdx_CDC" in all_branches): self.branches_dict["proton_dEdx_CDC_ln"] = np.log(self.branches_dict["proton_dEdx_CDC"]-1.)+4*self.branches_dict["p4_prot_pmag"] # if this quantity GREATER THAN than 2.25, accept as proton candidate
				if(self.mode=="3piq"):
					self.branches_dict["p4_pip_pmag"]   = np.sqrt( self.branches_dict["p4_pip_meas_px"]**2 + self.branches_dict["p4_pip_meas_py"]**2 + self.branches_dict["p4_pip_meas_pz"]**2)
					self.branches_dict["p4_pim_pmag"]   = np.sqrt( self.branches_dict["p4_pim_meas_px"]**2 + self.branches_dict["p4_pim_meas_py"]**2 + self.branches_dict["p4_pim_meas_pz"]**2)
					if("pip_dEdx_CDC" in all_branches): self.branches_dict["pip_dEdx_CDC_ln"] = np.log(self.branches_dict["pip_dEdx_CDC"]-6.2)+7*self.branches_dict["p4_pip_pmag"] # if this quantity LESS THAN than 3.0, accept as pi^+/- candidate
					if("pim_dEdx_CDC" in all_branches): self.branches_dict["pim_dEdx_CDC_ln"] = np.log(self.branches_dict["pim_dEdx_CDC"]-6.2)+7*self.branches_dict["p4_pim_pmag"] # if this quantity LESS THAN than 3.0, accept as pi^+/- candidate
		
	def CalcAdditionalBranchesPostCuts(self,get_Ephotons_sum):
		if(self.gen_diagnostic_hists):
			self.branches_dict["NCombos"]        = 1./self.branches_dict["FS_weight"]
			# For tracks
			self.branches_dict["p4_prot_kin_p"]  = np.sqrt( self.branches_dict["p4_prot_kin_px"]**2 + self.branches_dict["p4_prot_kin_py"]**2 + self.branches_dict["p4_prot_kin_pz"]**2)
			self.branches_dict["prot_theta_deg"] = np.arccos( self.branches_dict["p4_prot_kin_pz"]/self.branches_dict["p4_prot_kin_p"])*(180/3.14159)
			self.branches_dict["prot_phi_deg"]   = np.arctan2( self.branches_dict["p4_prot_kin_py"],self.branches_dict["p4_prot_kin_px"])*(180/3.14159)
			if(self.mode=="3piq"):
				self.branches_dict["p4_pip_kin_p"]  = np.sqrt( self.branches_dict["p4_pip_kin_px"]**2 + self.branches_dict["p4_pip_kin_py"]**2 + self.branches_dict["p4_pip_kin_pz"]**2)
				self.branches_dict["pip_theta_deg"] = np.arccos( self.branches_dict["p4_pip_kin_pz"]/self.branches_dict["p4_pip_kin_p"])*(180/3.14159)
				self.branches_dict["pip_phi_deg"]   = np.arctan2( self.branches_dict["p4_pip_kin_py"],self.branches_dict["p4_pip_kin_px"])*(180/3.14159)
				self.branches_dict["p4_pim_kin_p"]  = np.sqrt( self.branches_dict["p4_pim_kin_px"]**2 + self.branches_dict["p4_pim_kin_py"]**2 + self.branches_dict["p4_pim_kin_pz"]**2)
				self.branches_dict["pim_theta_deg"] = np.arccos( self.branches_dict["p4_pim_kin_pz"]/self.branches_dict["p4_pim_kin_p"])*(180/3.14159)
				self.branches_dict["pim_phi_deg"]   = np.arctan2( self.branches_dict["p4_pim_kin_py"],self.branches_dict["p4_pim_kin_px"])*(180/3.14159)
				self.branches_dict["NumLowThetaTracks"] =  np.where(self.branches_dict["pip_theta_deg"]<20.,1,0)
				self.branches_dict["NumLowThetaTracks"] += np.where(self.branches_dict["pim_theta_deg"]<20.,1,0)
				self.branches_dict["p4_pip_meas_p"]      = np.sqrt( self.branches_dict["p4_pip_meas_px"]**2 + self.branches_dict["p4_pip_meas_py"]**2 + self.branches_dict["p4_pip_meas_pz"]**2)
				self.branches_dict["pip_theta_meas_deg"] = np.arccos( self.branches_dict["p4_pip_meas_pz"]/self.branches_dict["p4_pip_meas__E"])*(180/3.14159)
		return

	# For organization's sake, all branches listed here in one place
	# # step options: "precut", "postcut"
	def PruneBranches(self,step):
		# Get list of branches in dict
		branchMemSizeMB=0
		for key in self.branches_dict: branchMemSizeMB+=sys.getsizeof(self.branches_dict[key])/(1024.)**2 # Megabytes		
		if(branchMemSizeMB<1500): print("Size of dictionary (MB) before step " + step + ": " + str(branchMemSizeMB))
		else:                     print("Size of dictionary (GB) before step " + step + ": " + str(branchMemSizeMB/1024.))
		# It's ok if any key specified is not found in branches_dict
		# Recall that to apply any cut (which we still will want to do after all pruning) 
		# #  Still need event, p4_beam__E, and x4_beam_t
		if(step=="precut"):
			# Cut study variables no longer needed (fine for general use, don't need to be in dict)
			branchesToPrune = ["eta_can_px","eta_can_py","eta_can_pz","eta_can__E","missing_px","missing_py","missing_pz","p4_pip_pmag","p4_pim_pmag","proton_dEdx_CDC","pip_dEdx_CDC","pim_dEdx_CDC"]
			for i in range(1,self.N_gammas+1): 
				branchesToPrune.extend(["p4_g"+str(i)+"_meas_px","p4_g"+str(i)+"_meas_py","p4_g"+str(i)+"_meas_pz"])
				branchesToPrune.extend(["p4_g"+str(i)+"_kin_px","p4_g"+str(i)+"_kin_py","p4_g"+str(i)+"_kin_pz"])
			for branch in branchesToPrune: 
				if(branch in self.branches_dict): del self.branches_dict[branch]
		elif(step=="postcut"):
			branchesToPrune = ["chi2_ndf_DiffFromMin","run"]
			for i in range(1,self.N_gammas+1): branchesToPrune.extend(["g"+str(i)+"_isFCAL","g"+str(i)+"_phi_deg_meas"])
			branchesToPrune.extend(["p4_prot_kin__E","p4_prot_kin_px", "p4_prot_kin_py", "p4_prot_kin_pz", "p4_prot_meas__E", "p4_prot_meas_px", "p4_prot_meas_py", "p4_prot_meas_pz","missing__E"])
			for branch in branchesToPrune: 
				if(branch in self.branches_dict): del self.branches_dict[branch]
		else:
			print("ERROR: unexpected option passed to PruneBranches " + step + " specified. Exiting...")
			sys.exit()
		branchMemSizeMB=0
		for key in self.branches_dict: branchMemSizeMB+=sys.getsizeof(self.branches_dict[key])/(1024.)**2 # Megabytes		
		print("Size of dictionary after step " + step + ": " + str(branchMemSizeMB))
		
		
	def CreateMesonSpecificHists(self,h_dict,h_dict_diagnostic,sample):
		h_dict["h_minus_t_"+sample]   = TH1F("h_minus_t_"+sample,"",self.NUM_MASS_BINS*5,0.,25.)
		for ebin in range(self.ebin_lo,self.ebin_hi+1): 
			h_dict["h2_eta_kin_ebin"+str(ebin)+"_"+sample] = TH2F("h2_eta_kin_ebin"+str(ebin)+"_"+sample,"",self.NUM_MASS_BINS,0.,1.,self.NUM_T_BINS,0.,self.NUM_T_BINS)
			h_dict["h_egamma_ebin"+str(ebin)+"_"+sample]   = TH1F("h_egamma_ebin"+str(ebin)+"_"+sample,"",self.NUM_MASS_BINS//4,3.,12.)
			# Distribution of photons and tracks needed for single particle systematics
			h_dict["h2_NFCAL_ebin"+str(ebin)+"_"+sample] = TH2F("h2_NFCAL_ebin"+str(ebin)+"_"+sample,"",11,0.,11,self.NUM_T_BINS,0.,self.NUM_T_BINS)
			if(self.mode=="3piq"): h_dict["h2_NLowThetaTr_ebin"+str(ebin)+"_"+sample] = TH2F("h2_NLowThetaTr_ebin"+str(ebin)+"_"+sample,"",11,0.,11,self.NUM_T_BINS,0.,self.NUM_T_BINS)
		if(self.gen_diagnostic_hists): self.CreateDiagnosticHists(h_dict_diagnostic,sample)
		if(self.do_bggen_study):       self.CreateBGGENHists(h_dict)
		if(self.do_cut_flow_study):    self.CreateCutFlowHists(h_dict,h_dict_diagnostic,sample)
		if(self.binning_type=="WCT"): 
			h_dict["h_cosTheta_cm_"+sample]               = TH1F("h_cosTheta_cm_"+sample,"",self.NUM_MASS_BINS*5,-1.,1.)
			for ebin in range(self.ebin_lo,self.ebin_hi+1): 
				h_dict["h_W_ebin"+str(ebin)+"_"+sample]   = TH1F("h_W_ebin"+str(ebin)+"_"+sample,"",self.NUM_MASS_BINS//4,2.5,6.5)
		if(sample=="MC"): h_dict["h_MC_lowT_over_runs"]     = TH1F("h_MC_lowT_over_runs","Number events reconstructed in #eta mass range (0.1<|t|<0.5 GeV^{2}, all E_{#gamma}); Run Number",10000,10000*int(self.run_digit),10000*int(self.run_digit)+10000)
		
	def CreateDiagnosticHists(self,h_dict,sample,ebin=3):
	
		e = str(ebin)
	
		# h_dict[""+sample] # Just for copy/pasting
		yaxis_hi = 2 if self.binning_type=="ET" else 1.1
		h_dict["h3_proton_p_t_E_"+sample]     = TH3F("h3_proton_p_t_E_"+sample,"",      300,0.,12.,125,0,25.,10,6.5,11.5)   # Max p: about 12 GeV/c
		h_dict["h3_gamma_E_t_E_"+sample]      = TH3F("h3_gamma_E_t_E_"+sample,"",    300,0.,12.,125,0,25.,10,6.5,11.5)   # Max E: about 12
		h_dict["h3_proton_theta_t_E_"+sample] = TH3F("h3_proton_theta_t_E_"+sample,"",  200,0.,80.,125,0,25.,10,6.5,11.5)   # Max theta: about 80 degrees
		h_dict["h3_gamma_theta_t_E_"+sample]  = TH3F("h3_gamma_theta_t_E_"+sample,"",   140,0.,140.,125,0,25.,10,6.5,11.5)   # Max theta: about 80 degrees?
		h_dict["h3_kinfit_chi2_"+sample]      = TH3F("h3_kinfit_chi2_"+sample,"",   200,0.,20.,125,0,25.,10,6.5,11.5)   # Max kinfit: about 80 degrees?

		h_dict["h_eta_mass_diag_ebin3_"+sample] = TH1F("h_eta_mass_diag_ebin3_"+sample,"#eta Candidates for Given Beam Photon; Num Candidates",self.NUM_MASS_BINS//5,0.,1.)
		h_dict["h_NCombos_ebin3_"+sample]       = TH1F("h_NCombos_ebin3_"+sample,"#eta Candidates for Given Beam Photon; Num Candidates",10,0.,10.)
		h_dict["h_EBeam_ebin3_"+sample]         = TH1F("h_EBeam_ebin3_"+sample,"Beam E",1200,0.,12.)
		h_dict["h_tdist_ebin3_"+sample]         = TH2F("h_tdist_ebin3_"+sample,"Mandelstam |t|",self.NUM_MASS_BINS//5,0.,2.0,self.NUM_T_BINS,0.,2.0)
		h_dict["h_prot_pmag_ebin3_"+sample]     = TH2F("h_prot_pmag_ebin3_"+sample,"Proton momentum (GeV/c, kinfit)",self.NUM_MASS_BINS,0.,2.,self.NUM_T_BINS,0.,yaxis_hi)
		h_dict["h_prot_theta_ebin3_"+sample]    = TH2F("h_prot_theta_ebin3_"+sample,"Proton #theta kinfit (degrees)",self.NUM_MASS_BINS,0.,180.,self.NUM_T_BINS,0.,yaxis_hi)
		h_dict["h_prot_phi_ebin3_"+sample]      = TH2F("h_prot_phi_ebin3_"+sample,"Proton #phi kinfit (degrees)",self.NUM_MASS_BINS,-180.,180.,self.NUM_T_BINS,0.,yaxis_hi)
		h_dict["h_gamma_E_ebin3_"+sample]       = TH2F("h_gamma_E_ebin3_"+sample,"Photon Energy (GeV, kinfit)",self.NUM_MASS_BINS,0.,10.,self.NUM_T_BINS,0.,yaxis_hi)
		h_dict["h_gamma_theta_ebin3_"+sample]   = TH2F("h_gamma_theta_ebin3_"+sample,"Photon #theta (degrees)",self.NUM_MASS_BINS*3,0.,60.,self.NUM_T_BINS,0.,yaxis_hi)
		h_dict["h_gamma_phi_ebin3_"+sample]     = TH2F("h_gamma_phi_ebin3_"+sample,"Photon #phi (degrees)",self.NUM_MASS_BINS,-180.,180.,self.NUM_T_BINS,0.,yaxis_hi)
		h_dict["h_vertex_measX_ebin3_"+sample]  = TH2F("h_vertex_measX_ebin3_"+sample,"Target X (cm)",self.NUM_MASS_BINS,-4.,4.,self.NUM_T_BINS,0.,yaxis_hi)
		h_dict["h_vertex_measY_ebin3_"+sample]  = TH2F("h_vertex_measY_ebin3_"+sample,"Target Y (cm)",self.NUM_MASS_BINS,-4.,4.,self.NUM_T_BINS,0.,yaxis_hi)
		h_dict["h_vertex_measZ_ebin3_"+sample]  = TH2F("h_vertex_measZ_ebin3_"+sample,"Target Z (cm)",self.NUM_MASS_BINS,0.,100.,self.NUM_T_BINS,0.,yaxis_hi)
		h_dict["h2_FCALOccup_ebin3lot_"+sample] = TH2F("h2_FCALOccup_ebin3lot_"+sample,"FCAL Shower (x,y), |t|<0.5 GeV^{2}",220,-110.,110.,220,-110.,110.)
		h_dict["h2_FCALOccup_ebin3tbin5_"+sample] = TH2F("h2_FCALOccup_ebin3tbin5_"+sample,"FCAL Shower (x,y), 0.5<|t|<0.6 GeV^{2}",220,-110.,110.,220,-110.,110.)
		
		h_dict["h_prot_thetaVSpmag_ebin3_tbin4andUnder_"+sample]  = TH2F("h_prot_thetaVSpmag_ebin3_tbin4andUnder_"+sample,"proton Momentum vs #theta (ebin3,tbin<=4);#theta (degrees);Momentum (GeV/c)",500,40.,90.,500,0.,2.)
		
		if(self.mode=="3piq"):
			h_dict["h_pip_pmag_ebin3_"+sample]  = TH2F("h_pip_pmag_ebin3_"+sample,"#pi^{+} momentum (GeV/c, kinfit)",self.NUM_MASS_BINS,0.,10.,self.NUM_T_BINS,0.,yaxis_hi)
			h_dict["h_pip_theta_ebin3_"+sample] = TH2F("h_pip_theta_ebin3_"+sample,"#pi^{+}  #theta kinfit (degrees)",self.NUM_MASS_BINS,0.,20.,self.NUM_T_BINS,0.,yaxis_hi)
			h_dict["h_pip_phi_ebin3_"+sample]   = TH2F("h_pip_phi_ebin3_"+sample,"#pi^{+}  #phi kinfit (degrees)",self.NUM_MASS_BINS,-180.,180.,self.NUM_T_BINS,0.,yaxis_hi)
			h_dict["h_pim_pmag_ebin3_"+sample]  = TH2F("h_pim_pmag_ebin3_"+sample,"#pi^{-}  momentum (GeV/c, kinfit)",self.NUM_MASS_BINS,0.,10.,self.NUM_T_BINS,0.,yaxis_hi)
			h_dict["h_pim_theta_ebin3_"+sample] = TH2F("h_pim_theta_ebin3_"+sample,"#pi^{-}  #theta kinfit (degrees)",self.NUM_MASS_BINS,0.,20.,self.NUM_T_BINS,0.,yaxis_hi)
			h_dict["h_pim_phi_ebin3_"+sample]   = TH2F("h_pim_phi_ebin3_"+sample,"#pi^{-}  #phi kinfit (degrees)",self.NUM_MASS_BINS,-180.,180.,self.NUM_T_BINS,0.,yaxis_hi)
			h_dict["h_pip_thetaVSpmag_ebin3_tbin4andUnder_"+sample]  = TH2F("h_pip_thetaVSpmag_ebin3_tbin4andUnder_"+sample,"#pi^{+} Momentum vs #theta (ebin3,tbin<=4);#theta (degrees);Momentum (GeV/c)",500,0.,30.,500,0.,10.)
			h_dict["h_pim_thetaVSpmag_ebin3_tbin4andUnder_"+sample]  = TH2F("h_pim_thetaVSpmag_ebin3_tbin4andUnder_"+sample,"#pi^{-} Momentum vs #theta (ebin3,tbin<=4);#theta (degrees);Momentum (GeV/c)",500,0.,30.,500,0.,10.)
		if(not self.UseFCALCut): 
			h_dict["h_gamma_theta_ebin9_"+sample] = TH2F("h_gamma_theta_ebin9_"+sample,"Photon #theta (degrees)",self.NUM_MASS_BINS*3,0.,60.,self.NUM_T_BINS,0.,yaxis_hi)
			h_dict["h_gamma_thetaMEAS_ebin9_"+sample] = TH2F("h_gamma_thetaMEAS_ebin9_"+sample,"Photon #theta (degrees)",self.NUM_MASS_BINS*3,0.,60.,self.NUM_T_BINS,0.,yaxis_hi)
		
	def CreateBGGENHists(self,h_dict):
		for ebin in range(self.ebin_lo,self.ebin_hi+1):
			h_dict["h2_eta_kin_ebin"+str(ebin)+"_bggen_incl"] = TH2F("h2_eta_kin_ebin"+str(ebin)+"_bggen_incl","",self.NUM_MASS_BINS,0.,1.,self.NUM_T_BINS,0.,self.NUM_T_BINS)
			h_dict["h2_eta_kin_ebin"+str(ebin)+"_bggen_excl"] = TH2F("h2_eta_kin_ebin"+str(ebin)+"_bggen_excl","",self.NUM_MASS_BINS,0.,1.,self.NUM_T_BINS,0.,self.NUM_T_BINS)
			h_dict["h2_eta_kin_ebin"+str(ebin)+"_bggen_nonEta"] = TH2F("h2_eta_kin_ebin"+str(ebin)+"_bggen_nonEta","",self.NUM_MASS_BINS,0.,1.,self.NUM_T_BINS,0.,self.NUM_T_BINS)
			
	def CreateCutFlowHists(self,h_dict,h_dict_diagnostic,sample):
		titleEnd_str = "ebins "+str(self.cut_flow_study_ebinLo)+"-"+str(self.cut_flow_study_ebinHi)+", tbins "+str(self.cut_flow_study_tbinLo)+"-"+str(self.cut_flow_study_tbinHi)
		yaxis_hi = 2 if self.binning_type=="ET" else 1.1
		
		# # Check that default cuts have been turned off...
		# PassedCutCheck=True
		# all_cut_vars = [cut_list[0] for cut_list in self.all_cut_list]
		# if("x4_prot_meas_z" in all_cut_vars and "p4_prot_pmag" in all_cut_vars and "chi2_ndf" in all_cut_vars):
		# 	print("WARNING! Found all the usual cuts in our cut list! This means you probably forgot to alter it for this particular instance. Check and rerun...")
		# 	print("ALL CUTS: " + str(self.all_cut_list))
		# 	sys.exit()
		
		h_dict["h_etaCutFlow_0_NoCuts_"+sample]     = TH1F("h_etaCutFlow_0_NoCuts_"+sample,"#eta Candidate Mass "+titleEnd_str+", inv mass (GeV)",self.NUM_MASS_BINS,0.,1.,)
		h_dict["h_etaCutFlow_1_ProtonBCAL_"+sample] = TH1F("h_etaCutFlow_1_ProtonBCAL_"+sample,"#eta Candidate Mass "+titleEnd_str+", inv mass (GeV)",self.NUM_MASS_BINS,0.,1.,)
		h_dict["h_etaCutFlow_2_PiqTOF_"+sample]     = TH1F("h_etaCutFlow_2_PiqTOF_"+sample,"#eta Candidate Mass "+titleEnd_str+", inv mass (GeV)",self.NUM_MASS_BINS,0.,1.,)
		h_dict["h_etaCutFlow_3_gamFCAL_"+sample]    = TH1F("h_etaCutFlow_3_gamFCAL_"+sample,"#eta Candidate Mass "+titleEnd_str+", inv mass (GeV)",self.NUM_MASS_BINS,0.,1.,)
		h_dict["h_etaCutFlow_4_otherPID_"+sample]   = TH1F("h_etaCutFlow_4_otherPID_"+sample,"#eta Candidate Mass "+titleEnd_str+", inv mass (GeV)",self.NUM_MASS_BINS,0.,1.,)
		h_dict["h_etaCutFlow_5_PIDhit_"+sample]     = TH1F("h_etaCutFlow_5_PIDhit_"+sample,"#eta Candidate Mass "+titleEnd_str+", inv mass (GeV)",self.NUM_MASS_BINS,0.,1.,)
		
		# Standard order that cuts are applied:
		# # DANAREST: FCAL shower/track veto-ing. dE/dx also???
		# # Step 0: min/max # particles, RF vote, measured inv. mass?
		# # Step 1: MM^2
		# # Step 2: Kinfit convergence
		# # Step 3: PID Timing cuts
		# # Step 4+ (DSelector level): kinfit chi^2, vertex z, proton momentum
		
		# Cut flow: PID Delta t, dE/dx, MM2, proton_pmag, chi2_ndf, beam energy, vertex z, fiducial_photon_theta
		
		
		# 1D distributions of variables, just before any cuts applied
		# THESE WILL GO INTO DIAGNOSTIC PLOTS!
		for det_sys in ["FCAL","BCAL","TOF","SC"]:
			h_dict_diagnostic["h_PIDT_proton_"+det_sys+"_"+sample]       = TH2F("h_PIDT_proton_"+det_sys+"_"+sample,"PID #Delta t, proton in "+ det_sys + " " +titleEnd_str+", PID #Delta t (ns)",500,-5.,5.,self.NUM_T_BINS,0.,yaxis_hi)
			if(det_sys=="FCAL" or det_sys == "BCAL"): h_dict_diagnostic["h_PIDT_gamma_"+det_sys+"_"+sample]        = TH2F("h_PIDT_gamma_"+det_sys+"_"+sample,"PID #Delta t, #gamma in "+ det_sys + " " +titleEnd_str+", PID #Delta t (ns)",500,-5.,5.,self.NUM_T_BINS,0.,yaxis_hi)
			if(self.mode=="3piq"): h_dict_diagnostic["h_PIDT_pip_"+det_sys+"_"+sample] = TH2F("h_PIDT_pip_"+det_sys+"_"+sample,"PID #Delta t, #pi^{+} in "+ det_sys + " " +titleEnd_str+", PID #Delta t (ns)",500,-5.,5.,self.NUM_T_BINS,0.,yaxis_hi)
			if(self.mode=="3piq"): h_dict_diagnostic["h_PIDT_pim_"+det_sys+"_"+sample] = TH2F("h_PIDT_pim_"+det_sys+"_"+sample,"PID #Delta t, #pi^{-} in "+ det_sys + " " +titleEnd_str+", PID #Delta t (ns)",500,-5.,5.,self.NUM_T_BINS,0.,yaxis_hi)
		h_dict_diagnostic["h_FCAL_DOCA_"+sample]       = TH2F("h_FCAL_DOCA_"+sample,"#eta Candidates "+titleEnd_str+", Photon/Track DOCA (cm);",500,0.,20.,self.NUM_T_BINS,0.,yaxis_hi)
		
		return
		
	def FillMesonSpecificHists(self,h_dict,h_dict_diagnostic,sample):
		gp.FillHistFromBranchDict(h_dict["h_minus_t_"+sample],self.branches_dict,"minus_t")
		if(self.binning_type=="WCT"): gp.FillHistFromBranchDict(h_dict["h_cosTheta_cm_"+sample],self.branches_dict,"cos_theta_cm")
		
		# Fill standard hists
		for ebin in range(self.ebin_lo,self.ebin_hi+1):
			print("Filling for histogram: " + h_dict["h2_eta_kin_ebin"+str(ebin)+"_"+sample].GetName())
			
			# No need to copy most of the branches over, so only use the ones specified here
			branches_to_keep = ["eta_mass_kin","NumFCALShowers","ebin","tbin"]
			if(self.binning_type=="WCT"): branches_to_keep.append("W")
			if(self.mode=="3piq" and "NumLowThetaTracks" in self.branches_dict): branches_to_keep.append("NumLowThetaTracks")
			
			branches_dict_thisEbin = gp.ApplyCutsReduceArrays(self.branches_dict,[["ebin","ebin == ",ebin]],branches_to_keep=branches_to_keep)
			gp.FillHistFromBranchDict(  h_dict["h_egamma_ebin"  +str(ebin)+"_"+sample],branches_dict_thisEbin,"p4_beam__E")
			gp.Fill2DHistFromBranchDict(h_dict["h2_eta_kin_ebin"+str(ebin)+"_"+sample],branches_dict_thisEbin,"eta_mass_kin","tbin")
			if(self.binning_type=="WCT"): gp.FillHistFromBranchDict(h_dict["h_W_ebin"+str(ebin)+"_"+sample],branches_dict_thisEbin,"W")
			
			# For determining number of FCAL/BCAL photons in eta mass window
			branches_dict_thisEbin_etaRange = gp.ApplyCutsReduceArrays(branches_dict_thisEbin,[["eta_mass_kin","eta_mass_kin > ",0.52],["eta_mass_kin","eta_mass_kin < ",0.58]])
			gp.Fill2DHistFromBranchDict(h_dict["h2_NFCAL_ebin"+str(ebin)+"_"+sample],branches_dict_thisEbin_etaRange,"NumFCALShowers","tbin")
			if(self.mode=="3piq"  and "NumLowThetaTracks" in self.branches_dict): gp.Fill2DHistFromBranchDict(h_dict["h2_NLowThetaTr_ebin"+str(ebin)+"_"+sample],branches_dict_thisEbin_etaRange,"NumLowThetaTracks","tbin")
			
			if(self.make_example_plots and ebin==3):
				print("WARNING: make_example_plots NOT FULLY IMPLEMENTED YET, exiting...")
				sys.exit()
				branches_dict_thistbin            = gp.ApplyCutsReduceArrays(branches_dict_thisEbin,[["tbin","tbin == ",4]])
				branches_dict_thistbin_intime     = gp.ApplyCutsReduceArrays(branches_dict_thistbin,[["accidweight","accidweight > ",0.],])
				branches_dict_thistbin_outoftime  = gp.ApplyCutsReduceArrays(branches_dict_thistbin,[["accidweight","accidweight < ",0.],])
				# Susan requested in-time and out-of-time hists shown separately
				gp.FillHistFromBranchDict(h_dict["h_eta_example_intime_ebin3_tbin4_"+sample],branches_dict_thistbin_intime,"eta_mass_kin")
				gp.FillHistFromBranchDict(h_dict["h_eta_example_outoftime_ebin3_tbin4_"+sample],branches_dict_thistbin_outoftime,"eta_mass_kin",DoAccidentalSub=False)
				gp.FillHistFromBranchDict(h_dict["h_eta_example_kin_ebin3_tbin4_"+sample],branches_dict_thistbin,"eta_mass_kin")
				gp.FillHistFromBranchDict(h_dict["h_eta_example_meas_ebin3_tbin4_"+sample],branches_dict_thistbin,"eta_mass_meas")
				del branches_dict_thistbin,branches_dict_thistbin_intime,branches_dict_thistbin_outoftime
			del branches_dict_thisEbin
		if(self.gen_diagnostic_hists): self.FillMesonSpecificHistsDiagnostic(h_dict_diagnostic,sample)
		if(self.do_cut_flow_study):    self.FillCutFlowHists(h_dict,h_dict_diagnostic,sample)
		if(self.do_bggen_study):       self.FillBGGENHists(h_dict)
		
		# Fill MC-specific hist for efficiency as a function of run number
		if(sample=="MC"):
			MC_EFFIC_LOW_T_CUTS = [
				["eta_mass_kin","eta_mass_kin > ",0.52],
				["eta_mass_kin","eta_mass_kin < ",0.58],
				["minus_t","minus_t < ",0.5],
			]
			branches_to_keep = ["run","eta_mass_kin"]
			branches_dict_MCeffic = gp.ApplyCutsReduceArrays(self.branches_dict,MC_EFFIC_LOW_T_CUTS,branches_to_keep=branches_to_keep)
			gp.FillHistFromBranchDict(h_dict["h_MC_lowT_over_runs"],branches_dict_MCeffic,"run")
		
		
		return
		
	def FillMesonSpecificHistsDiagnostic(self,h_dict_diagnostic,sample):

		all_branches = self.branches_dict.keys()

		# For copy/pasting...
		# gp.FillHistFromBranchDict(h_dict_diagnostic[""+sample], branches_dict_diagnosticHists,"variable")
		print("Filling diagnostic histograms...")
		ebin_diag=3 if self.binning_type=="ET" else 1
		if self.diag_bin_override>=0: ebin_diag=self.diag_bin_override

		# Now cut on eta inv. mass and |t|
		DIAGNOSTIC_PLOT_CUTS = [
			["eta_mass_kin","eta_mass_kin > ",0.52],
			["eta_mass_kin","eta_mass_kin < ",0.58],
		]

		branches_dict_diagnosticInit = gp.ApplyCutsReduceArrays(self.branches_dict,DIAGNOSTIC_PLOT_CUTS)

		# Fill 3D histograms
		gp.Fill3DHistFromBranchDict(h_dict_diagnostic["h3_proton_p_t_E_"+sample], branches_dict_diagnosticInit, "p4_prot_kin_p", "minus_t", "p4_beam__E")
		gp.Fill3DHistFromBranchDict(h_dict_diagnostic["h3_proton_theta_t_E_"+sample], branches_dict_diagnosticInit, "prot_theta_deg", "minus_t", "p4_beam__E")
		gp.Fill3DHistFromBranchDict(h_dict_diagnostic["h3_kinfit_chi2_"+sample], branches_dict_diagnosticInit, "chi2_ndf", "minus_t", "p4_beam__E")

		for i in range(1,self.N_gammas+1):
			branchname_gamE, branchname_gamtheta = "p4_g"+str(i)+"_kin__E",  "g"+str(i)+"_theta_deg"
			gp.Fill3DHistFromBranchDict(h_dict_diagnostic["h3_gamma_E_t_E_"+sample],     branches_dict_diagnosticInit, branchname_gamE,     "minus_t", "p4_beam__E")
			gp.Fill3DHistFromBranchDict(h_dict_diagnostic["h3_gamma_theta_t_E_"+sample], branches_dict_diagnosticInit, branchname_gamtheta, "minus_t", "p4_beam__E")


		if(self.diagnostic_hists_sideband):
			del DIAGNOSTIC_PLOT_CUTS
			DIAGNOSTIC_PLOT_CUTS = [
				["eta_mass_kin","eta_mass_kin > ",0.6],
				["eta_mass_kin","eta_mass_kin < ",0.65],
			]
		
		y_axis_var=""
		if(self.binning_type=="ET"):  
			DIAGNOSTIC_PLOT_CUTS.append(["minus_t","minus_t < ",2.0])
			if(ebin_diag==0): DIAGNOSTIC_PLOT_CUTS.append(["ebin","ebin < ",1.])
			if(ebin_diag!=0): DIAGNOSTIC_PLOT_CUTS.append(["ebin","ebin == ",ebin_diag])
			y_axis_var="minus_t"
		if(self.binning_type=="WCT"): 
			DIAGNOSTIC_PLOT_CUTS.append(["ebin","ebin <= ",ebin_diag])
			y_axis_var="cos_theta_cm"

		print("DIAGNOSTIC HISTS WITH EBIN "+str(ebin_diag))
		print("DIAGNOSTIC HISTS WITH EBIN "+str(ebin_diag))
		print("DIAGNOSTIC HISTS WITH EBIN "+str(ebin_diag))
		print("DIAGNOSTIC HISTS WITH EBIN "+str(ebin_diag))
		print("DIAGNOSTIC HISTS WITH EBIN "+str(ebin_diag))
		print("DIAGNOSTIC HISTS WITH EBIN "+str(ebin_diag))

		branches_dict_diagnosticHists = gp.ApplyCutsReduceArrays(branches_dict_diagnosticInit,DIAGNOSTIC_PLOT_CUTS)
		# Event/combo info
		branches_dict_diagnosticHists_intime = gp.ApplyCutsReduceArrays(branches_dict_diagnosticHists,[["accidweight","accidweight > ",0.],])
		gp.FillHistFromBranchDict(  h_dict_diagnostic["h_eta_mass_diag_ebin3_"+sample],branches_dict_diagnosticHists_intime,"eta_mass_kin")	# If NCombos=4, we'd fill with weight 0.25 four times with this call.
		gp.FillHistFromBranchDict(  h_dict_diagnostic["h_NCombos_ebin3_"+sample],branches_dict_diagnosticHists_intime,"NCombos",DoAccidentalSub=False,DoFSWeighting=True)	# If NCombos=4, we'd fill with weight 0.25 four times with this call.
		gp.Fill2DHistFromBranchDict(h_dict_diagnostic["h_tdist_ebin3_"+sample],branches_dict_diagnosticHists,"minus_t",y_axis_var)	# If NCombos=4, we'd fill with weight 0.25 four times with this call.
		gp.FillHistFromBranchDict(  h_dict_diagnostic["h_EBeam_ebin3_"+sample],branches_dict_diagnosticHists,"p4_beam__E",y_axis_var)	# If NCombos=4, we'd fill with weight 0.25 four times with this call.
		gp.Fill2DHistFromBranchDict(h_dict_diagnostic["h_vertex_measX_ebin3_"+sample],branches_dict_diagnosticHists,"x4_prot_meas_x",y_axis_var)	# If NCombos=4, we'd fill with weight 0.25 four times with this call.
		gp.Fill2DHistFromBranchDict(h_dict_diagnostic["h_vertex_measY_ebin3_"+sample],branches_dict_diagnosticHists,"x4_prot_meas_y",y_axis_var)	# If NCombos=4, we'd fill with weight 0.25 four times with this call.
		gp.Fill2DHistFromBranchDict(h_dict_diagnostic["h_vertex_measZ_ebin3_"+sample],branches_dict_diagnosticHists,"x4_prot_meas_z",y_axis_var)	# If NCombos=4, we'd fill with weight 0.25 four times with this call.
		# Fill for photons
		for i in range(1,self.N_gammas+1):
			gp.Fill2DHistFromBranchDict(h_dict_diagnostic["h_gamma_E_ebin3_"+sample],branches_dict_diagnosticHists,"p4_g"+str(i)+"_kin__E",y_axis_var)
			gp.Fill2DHistFromBranchDict(h_dict_diagnostic["h_gamma_theta_ebin3_"+sample],branches_dict_diagnosticHists,"g"+str(i)+"_theta_deg",y_axis_var)
			gp.Fill2DHistFromBranchDict(h_dict_diagnostic["h_gamma_phi_ebin3_"+sample],branches_dict_diagnosticHists,"g"+str(i)+"_phi_deg",y_axis_var)
		# Fill for tracks
		gp.Fill2DHistFromBranchDict(h_dict_diagnostic["h_prot_pmag_ebin3_"+sample],branches_dict_diagnosticHists,"p4_prot_kin_p",y_axis_var)
		gp.Fill2DHistFromBranchDict(h_dict_diagnostic["h_prot_theta_ebin3_"+sample],branches_dict_diagnosticHists,"prot_theta_deg",y_axis_var)
		gp.Fill2DHistFromBranchDict(h_dict_diagnostic["h_prot_phi_ebin3_"+sample],branches_dict_diagnosticHists,"prot_phi_deg",y_axis_var)
		
		# Now for low |t| region specifically
		if(self.binning_type=="ET"):
			LOWT_CUT = [
				["tbin","tbin < ",4.01],
			]
			branches_dict_diagnosticLowT = gp.ApplyCutsReduceArrays(branches_dict_diagnosticHists,LOWT_CUT)
			gp.Fill2DHistFromBranchDict(h_dict_diagnostic["h_prot_thetaVSpmag_ebin3_tbin4andUnder_"+sample],branches_dict_diagnosticLowT,"prot_theta_deg","p4_prot_kin_p")

		if(self.mode=="3piq"):
			if("pip_PIDT_TOF" in all_branches): gp.Fill2DHistFromBranchDict(h_dict_diagnostic["h_PIDT_pip_TOF_"+sample],branches_dict_diagnosticHists,"pip_PIDT_TOF",y_axis_var)
			if("pim_PIDT_TOF" in all_branches): gp.Fill2DHistFromBranchDict(h_dict_diagnostic["h_PIDT_pim_TOF_"+sample],branches_dict_diagnosticHists,"pim_PIDT_TOF",y_axis_var)


			gp.Fill2DHistFromBranchDict(h_dict_diagnostic["h_pip_pmag_ebin3_"+sample],branches_dict_diagnosticHists,"p4_pip_kin_p",y_axis_var)
			gp.Fill2DHistFromBranchDict(h_dict_diagnostic["h_pip_theta_ebin3_"+sample],branches_dict_diagnosticHists,"pip_theta_deg",y_axis_var)
			gp.Fill2DHistFromBranchDict(h_dict_diagnostic["h_pip_phi_ebin3_"+sample],branches_dict_diagnosticHists,"pip_phi_deg",y_axis_var)
			gp.Fill2DHistFromBranchDict(h_dict_diagnostic["h_pim_pmag_ebin3_"+sample],branches_dict_diagnosticHists,"p4_pim_kin_p",y_axis_var)
			gp.Fill2DHistFromBranchDict(h_dict_diagnostic["h_pim_theta_ebin3_"+sample],branches_dict_diagnosticHists,"pim_theta_deg",y_axis_var)
			gp.Fill2DHistFromBranchDict(h_dict_diagnostic["h_pim_phi_ebin3_"+sample],branches_dict_diagnosticHists,"pim_phi_deg",y_axis_var)
			# Now for low |t| region specifically
			if(self.binning_type=="ET"):
				LOWT_CUT = [
					["tbin","tbin < ",4.01],
				]
				branches_dict_diagnosticLowT = gp.ApplyCutsReduceArrays(branches_dict_diagnosticHists,LOWT_CUT)
				# gp.Fill2DHistFromBranchDict(h_dict_diagnostic["h_prot_thetaVSpmag_ebin3_tbin4andUnder_"+sample],branches_dict_diagnosticLowT,"prot_theta_deg","p4_prot_kin_p")
				gp.Fill2DHistFromBranchDict(h_dict_diagnostic["h_pip_thetaVSpmag_ebin3_tbin4andUnder_"+sample],branches_dict_diagnosticLowT,"pip_theta_deg","p4_pip_kin_p")
				gp.Fill2DHistFromBranchDict(h_dict_diagnostic["h_pim_thetaVSpmag_ebin3_tbin4andUnder_"+sample],branches_dict_diagnosticLowT,"pim_theta_deg","p4_pim_kin_p")
				del branches_dict_diagnosticLowT
		# For demonstrating FCAL occupancy with no theta cut, use highest E bin
		if(not self.UseFCALCut):
			DIAGNOSTIC_PLOT_CUTS[-1][-1]=9 # Change ebin to cut on
			branches_dict_diagnosticHists = gp.ApplyCutsReduceArrays(self.branches_dict,DIAGNOSTIC_PLOT_CUTS)
			for i in range(1,self.N_gammas+1): gp.Fill2DHistFromBranchDict(h_dict_diagnostic["h_gamma_theta_ebin9_"+sample],branches_dict_diagnosticHists,"g"+str(i)+"_theta_deg",y_axis_var)
			for i in range(1,self.N_gammas+1): gp.Fill2DHistFromBranchDict(h_dict_diagnostic["h_gamma_thetaMEAS_ebin9_"+sample],branches_dict_diagnosticHists,"g"+str(i)+"_theta_deg_meas",y_axis_var)
		# FCAL Occupancy: needs additional cut requiring photon to be in FCAL
		if(self.binning_type=="ET"):
			for i in range(1,self.N_gammas+1):
				branches_to_keep          = ["x4_g"+str(i)+"_shower_x","x4_g"+str(i)+"_shower_y","x4_g"+str(i)+"_shower_z"]
				branches_dict_gammaInFCAL_lowT = gp.ApplyCutsReduceArrays(branches_dict_diagnosticHists,[["x4_g"+str(i)+"_shower_z","x4_g"+str(i)+"_shower_z > ",600],["tbin","tbin < ",4.01]],branches_to_keep=branches_to_keep)
				branches_dict_gammaInFCAL_hiT  = gp.ApplyCutsReduceArrays(branches_dict_diagnosticHists,[["x4_g"+str(i)+"_shower_z","x4_g"+str(i)+"_shower_z > ",600],["tbin","tbin > ",4.99],["tbin","tbin < ",5.01]] ,branches_to_keep=branches_to_keep)
				gp.Fill2DHistFromBranchDict(h_dict_diagnostic["h2_FCALOccup_ebin3lot_"+sample],  branches_dict_gammaInFCAL_lowT,"x4_g"+str(i)+"_shower_x","x4_g"+str(i)+"_shower_y")
				gp.Fill2DHistFromBranchDict(h_dict_diagnostic["h2_FCALOccup_ebin3tbin5_"+sample],branches_dict_gammaInFCAL_hiT, "x4_g"+str(i)+"_shower_x","x4_g"+str(i)+"_shower_y")
				del branches_dict_gammaInFCAL_lowT, branches_dict_gammaInFCAL_hiT


		# Final cleanup diagnostic hists section
		del branches_dict_diagnosticHists,branches_dict_diagnosticHists_intime
		
	def FillCutFlowHists(self,h_dict,h_dict_diagnostic,sample):

		hasFCAL_DOCA  = True if self.mode=="3piq" and "g1_FCAL_DOCA" in self.branches_dict.keys() else False

		# Add branches to list that we'll save
		branches_to_keep = ["eta_mass_kin","ebin","tbin","minus_t"]
		if hasFCAL_DOCA: branches_to_keep.extend(["g1_FCAL_DOCA","g2_FCAL_DOCA",])

		# # Now add PID delta T stuff
		det_systems   = ["FCAL","BCAL","TOF","SC"]
		particle_list = ["proton"] + ["g"+str(i) for i in range(1,self.N_gammas+1)]
		if(self.mode=="3piq"): particle_list.extend(["pip","pim"])
		for p in particle_list:
			for det_sys in det_systems:
				if(p[0]=="g" and "CAL" not in det_sys): continue # This skips "SC" and "TOF" for photons
				bname = p+"_PIDT_"+det_sys  # e.g. proton_PIDT_BCAL
				branches_to_keep.append(bname)
					
		# Now cut on eta inv. mass and |t|, get branches
		CutFlowHists_CUTS = [
			["eta_mass_kin","eta_mass_kin > ",0.52],
			["eta_mass_kin","eta_mass_kin < ",0.58],
			["ebin","ebin > ",float(self.cut_flow_study_ebinLo)-0.001],
			["ebin","ebin < ",float(self.cut_flow_study_ebinHi)+0.001],
			["tbin","tbin > ",float(self.cut_flow_study_tbinLo)-0.001],
			["tbin","tbin < ",float(self.cut_flow_study_tbinHi)+0.001],
		]
		if(self.diagnostic_hists_sideband):
			CutFlowHists_CUTS[0]=["eta_mass_kin","eta_mass_kin > ",0.6]
			CutFlowHists_CUTS[1]=["eta_mass_kin","eta_mass_kin < ",0.65]
		
		
		branches_dict_CutFlowHists = gp.ApplyCutsReduceArrays(self.branches_dict,CutFlowHists_CUTS,branches_to_keep=branches_to_keep)
		
		# Calc PID hit branch, then do other stuff
		for p in particle_list:
			branches_dict_CutFlowHists[p+"_PIDsum"]      = np.zeros(len(branches_dict_CutFlowHists["ebin"]))
			for det_sys in det_systems:
				if(p[0]=="g" and "CAL" not in det_sys): continue # This skips "SC" and "TOF" for photons
				bname = p+"_PIDT_"+det_sys  # e.g. proton_PIDT_BCAL
				branches_dict_CutFlowHists[p+"_PIDsum"] += branches_dict_CutFlowHists[bname]
				branches_dict_CutFlowHists[bname][branches_dict_CutFlowHists[bname] <-999.] += 1000. # A[A < 0] += 5
			branches_dict_CutFlowHists[p+"_hasPIDhit"]   = np.where(branches_dict_CutFlowHists[p+"_PIDsum"]>-3000.,1,0)
		
		
		# FILL DIAGNOSTIC HISTS
		# # Naming example: "h_PIDT_proton_TOF_DATA"
		for p in particle_list:
			for det_sys in det_systems:
				if(p[0]=="g" and "CAL" not in det_sys): continue # This skips "SC" and "TOF" for photons
				bname = p+"_PIDT_"+det_sys  # e.g. proton_PIDT_BCAL
				hist_string = "h_PIDT_"+p+"_"+det_sys+"_"+sample
				if(p[0]=="g"): hist_string="h_PIDT_gamma_"+det_sys+"_"+sample
				if(hist_string not in h_dict_diagnostic):
					print("WARNING!!!! could not find histogram for " +hist_strin+ " skipping... ")
					continue
				gp.Fill2DHistFromBranchDict(h_dict_diagnostic[hist_string],branches_dict_CutFlowHists,bname,"minus_t")	# If NCombos=4, we'd fill with weight 0.25 four times with this call.
		if(hasFCAL_DOCA):
			gp.Fill2DHistFromBranchDict(h_dict_diagnostic["h_FCAL_DOCA_"+sample],branches_dict_CutFlowHists,"g1_FCAL_DOCA","minus_t")	# If NCombos=4, we'd fill with weight 0.25 four times with this call.
			gp.Fill2DHistFromBranchDict(h_dict_diagnostic["h_FCAL_DOCA_"+sample],branches_dict_CutFlowHists,"g2_FCAL_DOCA","minus_t")	# If NCombos=4, we'd fill with weight 0.25 four times with this call.
		
	
		# next onto filling eta mass after each cut....
		# pi+/-  really doesn't care about subsystems other than TOF
		# proton really doesn't care about subsystems other than BCAL
		# gammas care mostly about FCAL, but also BCAL
		# So let's go in order:
		gp.FillHistFromBranchDict(h_dict["h_etaCutFlow_0_NoCuts_"+sample],branches_dict_CutFlowHists,"eta_mass_kin")
		# # 1. Protons in BCAL
		branches_dict_CutFlowHists = gp.ApplyCutsReduceArrays(self.branches_dict,[["proton_PIDT_BCAL","proton_PIDT_BCAL lt_abs ",0.,1.]])
		gp.FillHistFromBranchDict(h_dict["h_etaCutFlow_0_NoCuts_"+sample],branches_dict_CutFlowHists,"eta_mass_kin")
		# # 2. Pions in TOF
		branches_dict_CutFlowHists = gp.ApplyCutsReduceArrays(self.branches_dict,[["pip_PIDT_TOF","pip_PIDT_TOF lt_abs ",0.,0.5],["pim_PIDT_TOF","pim_PIDT_TOF lt_abs ",0.,0.5]])
		gp.FillHistFromBranchDict(h_dict["h_etaCutFlow_0_NoCuts_"+sample],branches_dict_CutFlowHists,"eta_mass_kin")
		# # 3. Gammas in FCAL
		cut_list = []
		for i in range(1,self.N_gammas+1): cut_list.append(["g"+str(i)+"_PIDT_FCAL","g"+str(i)+"_PIDT_FCAL lt_abs ",0.,2.5])
		branches_dict_CutFlowHists = gp.ApplyCutsReduceArrays(self.branches_dict,cut_list)
		gp.FillHistFromBranchDict(h_dict["h_etaCutFlow_0_NoCuts_"+sample],branches_dict_CutFlowHists,"eta_mass_kin")
		# # # 4. Everything else PID Delta t
		# branches_dict_CutFlowHists = gp.ApplyCutsReduceArrays(self.branches_dict,[["proton_PIDT_BCAL","proton_PIDT_BCAL lt_abs ",0.,1.]])
		# gp.FillHistFromBranchDict(h_dict["h_etaCutFlow_0_NoCuts_"+sample],branches_dict_CutFlowHists,"eta_mass_kin")
		# # # 5. Cut no PID events
		# branches_dict_CutFlowHists = gp.ApplyCutsReduceArrays(self.branches_dict,[["proton_PIDT_BCAL","proton_PIDT_BCAL lt_abs ",0.,1.]])
		# gp.FillHistFromBranchDict(h_dict["h_etaCutFlow_0_NoCuts_"+sample],branches_dict_CutFlowHists,"eta_mass_kin")

		del branches_dict_CutFlowHists
		return

	def FillBGGENHists(self,h_dict):
		for ebin in range(self.ebin_lo,self.ebin_hi+1):
			# I checked that case0 + case1 + case2 adds up to full length
			branches_dict_thisEbin       = gp.ApplyCutsReduceArrays(self.branches_dict,[["ebin","ebin == ",ebin]])
			branches_dict_thisEbin_EXCL  = gp.ApplyCutsReduceArrays(branches_dict_thisEbin,[["MC_TopologyType","MC_TopologyType == ",0]])
			branches_dict_thisEbin_INCL  = gp.ApplyCutsReduceArrays(branches_dict_thisEbin,[["MC_TopologyType","MC_TopologyType == ",1]])
			branches_dict_thisEbin_OTHER = gp.ApplyCutsReduceArrays(branches_dict_thisEbin,[["MC_TopologyType","MC_TopologyType == ",2]])
			gp.Fill2DHistFromBranchDict(h_dict["h2_eta_kin_ebin"+str(ebin)+"_bggen_excl"],branches_dict_thisEbin_EXCL,"eta_mass_kin","tbin")
			gp.Fill2DHistFromBranchDict(h_dict["h2_eta_kin_ebin"+str(ebin)+"_bggen_incl"],branches_dict_thisEbin_INCL,"eta_mass_kin","tbin")
			gp.Fill2DHistFromBranchDict(h_dict["h2_eta_kin_ebin"+str(ebin)+"_bggen_nonEta"],branches_dict_thisEbin_OTHER,"eta_mass_kin","tbin")
			del branches_dict_thisEbin, branches_dict_thisEbin_EXCL, branches_dict_thisEbin_INCL, branches_dict_thisEbin_OTHER
		
	# Fit MC hist for initial parameters, then fit data histogram
	def FitXSecHistsOneBin(self,h_data,h_MC,ebin,tbin,df):
		MC_parms = self.FitMCHistGetParms(h_MC)
		data_int = jzHistogramSumIntegral(h_data,0.52,0.58)
		print("ebin,tbin: " + str(ebin) + ", " + str(tbin) + " sum in range: " + str(data_int))
		
		df_tmp = pd.DataFrame()
		
		rebin_factor = 2
		if(self.binning_type=="ET" and tbin>=self.NUM_T_BINS_LT):           rebin_factor=10
		if(self.binning_type=="WCT" and (tbin<=17 or "3pi" in self.mode) ): rebin_factor=10
		
		# Fit data hist (MAGIC HAPPENS HERE)
		if(self.binning_type=="ET"): 
			if(tbin<self.NUM_T_BINS_LT): df_tmp=self.FitDataHist(h_data,MC_parms,fix_to_MC_parms=False,rebin_factor=rebin_factor)
			else:                        df_tmp=self.FitDataHist(h_data,MC_parms,fix_to_MC_parms=True,rebin_factor=rebin_factor)
		if(self.binning_type=="WCT"):
			if(tbin<=17 or "3pi" in self.mode): df_tmp=self.FitDataHist(h_data,MC_parms,fix_to_MC_parms=True,rebin_factor=rebin_factor)
			else:                               df_tmp=self.FitDataHist(h_data,MC_parms,fix_to_MC_parms=False,rebin_factor=rebin_factor)
		
		# Fill out some more rows in temp df
		df_tmp["totbin"]    = ebin*100+tbin
		df_tmp["ebin"]      = ebin
		df_tmp["tbin"]      = tbin
		df_tmp["N_MCrecon"] = MC_parms["Case1_SigYield"][0]
		df_tmp["tbinLo"],df_tmp["tbinHi"] = self.TBINS_LO[ebin][tbin],self.TBINS_HI[ebin][tbin]
		if(self.mode=="gg"): df_tmp["YieldAboveSig"] = jzHistogramSumIntegral(h_data,0.5999,1.0001)
		else:                df_tmp["YieldAboveSig"] = 0.

		# Add all fits from this bin to total results
		df = pd.concat([df,df_tmp],ignore_index=True,sort=False)
		
		return df


	def FitDataHist(self,h,MC_parms,fix_to_MC_parms,rebin_factor=2):
		
		gROOT.Reset()
		curr_title=self.HISTTITLE_DICT[self.mode].split(";")[1]
		curr_canvas_tag = self.mode+"_"+self.run+"_"+self.tag
		
		this_var_name = "nominal"
		this_parm_list=self.GetFitInitList(self.mode,MC_parms,fix_to_MC_parms)
		jzFitter = jz2GausFit(self.CXX_DIR,rebin_factor,canvas_tag=curr_canvas_tag,xaxis_title=curr_title,*self.GetFitRangesByMode())
		if(self.save_fit_lvl>0): jzFitter.SetSaveHists(self.save_fit_lvl,self.FITTED_HIST_PLOTDIR)
		results_dict = jzFitter.FitHist3Cases(h,this_parm_list,this_var_name)
		gROOT.Reset()
		del jzFitter,this_parm_list
		
		# print "ALL DATA RESULTS: "
		# for key in results_dict: print key + " " + str(results_dict[key])
		# sys.exit()
		
		if(self.FIT_ALL_VARIATIONS):
		
			this_var_name = "2x_bins"
			this_parm_list=self.GetFitInitList(self.mode,MC_parms,fix_to_MC_parms)
			jzFitter = jz2GausFit(self.CXX_DIR,rebin_factor/2,canvas_tag=curr_canvas_tag,xaxis_title=curr_title,*self.GetFitRangesByMode())
			if(self.save_fit_lvl>0): jzFitter.SetSaveHists(self.save_fit_lvl,self.FITTED_HIST_PLOTDIR)
			results_tmp = jzFitter.FitHist3Cases(h,this_parm_list,this_var_name)
			if(results_tmp!=None):
				for key in results_dict: results_dict[key].extend( results_tmp[key] ) # key=string, val=list. Add to results
			gROOT.Reset()
			del jzFitter,this_parm_list
			
			this_var_name = "0.5x_bins"
			this_parm_list=self.GetFitInitList(self.mode,MC_parms,fix_to_MC_parms)
			jzFitter = jz2GausFit(self.CXX_DIR,rebin_factor*2,canvas_tag=curr_canvas_tag,xaxis_title=curr_title,*self.GetFitRangesByMode())
			if(self.save_fit_lvl>0): jzFitter.SetSaveHists(self.save_fit_lvl,self.FITTED_HIST_PLOTDIR)
			results_tmp = jzFitter.FitHist3Cases(h,this_parm_list,this_var_name)
			if(results_tmp!=None):
				for key in results_dict: results_dict[key].extend( results_tmp[key] ) # key=string, val=list. Add to results
			gROOT.Reset()
			del jzFitter,this_parm_list
			
			if(rebin_factor==2): this_var_name = "0.4x_bins"
			else:                this_var_name = "0.25x_bins"
			curr_rebin = 5 if (rebin_factor==2) else 20
			this_parm_list=self.GetFitInitList(self.mode,MC_parms,fix_to_MC_parms)
			jzFitter = jz2GausFit(self.CXX_DIR,curr_rebin,canvas_tag=curr_canvas_tag,xaxis_title=curr_title,*self.GetFitRangesByMode())
			if(self.save_fit_lvl>0): jzFitter.SetSaveHists(self.save_fit_lvl,self.FITTED_HIST_PLOTDIR)
			results_tmp = jzFitter.FitHist3Cases(h,this_parm_list,this_var_name)
			if(results_tmp!=None):
				for key in results_dict: results_dict[key].extend( results_tmp[key] ) # key=string, val=list. Add to results
			gROOT.Reset()
			del jzFitter,this_parm_list
			
			
			# Modifications to fit ranges
			this_var_name = "LoRange5MeV_narrower"
			this_parm_list=self.GetFitInitList(self.mode,MC_parms,fix_to_MC_parms)
			fitRangeLo,fitRangeHi,sigRejectRangeLo,sigRejectRangeHi,sigIntegrateRangeLo,sigIntegrateRangeHi = self.GetFitRangesByMode()
			fitRangeLo=fitRangeLo+0.005
			jzFitter = jz2GausFit(self.CXX_DIR,rebin_factor,fitRangeLo,fitRangeHi,sigRejectRangeLo,sigRejectRangeHi,sigIntegrateRangeLo,sigIntegrateRangeHi,canvas_tag=curr_canvas_tag,xaxis_title=curr_title)
			if(self.save_fit_lvl>0): jzFitter.SetSaveHists(self.save_fit_lvl,self.FITTED_HIST_PLOTDIR)
			results_tmp = jzFitter.FitHist3Cases(h,this_parm_list,this_var_name)
			if(results_tmp!=None):
				for key in results_dict: results_dict[key].extend( results_tmp[key] ) # key=string, val=list. Add to results
			gROOT.Reset()
			del jzFitter,this_parm_list
			this_var_name = "LoRange10MeV_narrower"
			this_parm_list=self.GetFitInitList(self.mode,MC_parms,fix_to_MC_parms)
			fitRangeLo,fitRangeHi,sigRejectRangeLo,sigRejectRangeHi,sigIntegrateRangeLo,sigIntegrateRangeHi = self.GetFitRangesByMode()
			fitRangeLo=fitRangeLo+0.010
			jzFitter = jz2GausFit(self.CXX_DIR,rebin_factor,fitRangeLo,fitRangeHi,sigRejectRangeLo,sigRejectRangeHi,sigIntegrateRangeLo,sigIntegrateRangeHi,canvas_tag=curr_canvas_tag,xaxis_title=curr_title)
			if(self.save_fit_lvl>0): jzFitter.SetSaveHists(self.save_fit_lvl,self.FITTED_HIST_PLOTDIR)
			results_tmp = jzFitter.FitHist3Cases(h,this_parm_list,this_var_name)
			if(results_tmp!=None):
				for key in results_dict: results_dict[key].extend( results_tmp[key] ) # key=string, val=list. Add to results
			gROOT.Reset()
			del jzFitter,this_parm_list
			this_var_name = "LoRange15MeV_narrower"
			this_parm_list=self.GetFitInitList(self.mode,MC_parms,fix_to_MC_parms)
			fitRangeLo,fitRangeHi,sigRejectRangeLo,sigRejectRangeHi,sigIntegrateRangeLo,sigIntegrateRangeHi = self.GetFitRangesByMode()
			fitRangeLo=fitRangeLo+0.015
			jzFitter = jz2GausFit(self.CXX_DIR,rebin_factor,fitRangeLo,fitRangeHi,sigRejectRangeLo,sigRejectRangeHi,sigIntegrateRangeLo,sigIntegrateRangeHi,canvas_tag=curr_canvas_tag,xaxis_title=curr_title)
			if(self.save_fit_lvl>0): jzFitter.SetSaveHists(self.save_fit_lvl,self.FITTED_HIST_PLOTDIR)
			results_tmp = jzFitter.FitHist3Cases(h,this_parm_list,this_var_name)
			if(results_tmp!=None):
				for key in results_dict: results_dict[key].extend( results_tmp[key] ) # key=string, val=list. Add to results
			gROOT.Reset()
			del jzFitter,this_parm_list
			this_var_name = "LoRange20MeV_narrower"
			this_parm_list=self.GetFitInitList(self.mode,MC_parms,fix_to_MC_parms)
			fitRangeLo,fitRangeHi,sigRejectRangeLo,sigRejectRangeHi,sigIntegrateRangeLo,sigIntegrateRangeHi = self.GetFitRangesByMode()
			fitRangeLo=fitRangeLo+0.020
			jzFitter = jz2GausFit(self.CXX_DIR,rebin_factor,fitRangeLo,fitRangeHi,sigRejectRangeLo,sigRejectRangeHi,sigIntegrateRangeLo,sigIntegrateRangeHi,canvas_tag=curr_canvas_tag,xaxis_title=curr_title)
			if(self.save_fit_lvl>0): jzFitter.SetSaveHists(self.save_fit_lvl,self.FITTED_HIST_PLOTDIR)
			results_tmp = jzFitter.FitHist3Cases(h,this_parm_list,this_var_name)
			if(results_tmp!=None):
				for key in results_dict: results_dict[key].extend( results_tmp[key] ) # key=string, val=list. Add to results
			gROOT.Reset()
			del jzFitter,this_parm_list
			this_var_name = "HiRange5MeV_narrower"
			this_parm_list=self.GetFitInitList(self.mode,MC_parms,fix_to_MC_parms)
			fitRangeLo,fitRangeHi,sigRejectRangeLo,sigRejectRangeHi,sigIntegrateRangeLo,sigIntegrateRangeHi = self.GetFitRangesByMode()
			fitRangeHi=fitRangeHi-0.005
			jzFitter = jz2GausFit(self.CXX_DIR,rebin_factor,fitRangeLo,fitRangeHi,sigRejectRangeLo,sigRejectRangeHi,sigIntegrateRangeLo,sigIntegrateRangeHi,canvas_tag=curr_canvas_tag,xaxis_title=curr_title)
			if(self.save_fit_lvl>0): jzFitter.SetSaveHists(self.save_fit_lvl,self.FITTED_HIST_PLOTDIR)
			results_tmp = jzFitter.FitHist3Cases(h,this_parm_list,this_var_name)
			if(results_tmp!=None):
				for key in results_dict: results_dict[key].extend( results_tmp[key] ) # key=string, val=list. Add to results
			gROOT.Reset()
			del jzFitter,this_parm_list
			this_var_name = "HiRange10MeV_narrower"
			this_parm_list=self.GetFitInitList(self.mode,MC_parms,fix_to_MC_parms)
			fitRangeLo,fitRangeHi,sigRejectRangeLo,sigRejectRangeHi,sigIntegrateRangeLo,sigIntegrateRangeHi = self.GetFitRangesByMode()
			fitRangeHi=fitRangeHi-0.010
			jzFitter = jz2GausFit(self.CXX_DIR,rebin_factor,fitRangeLo,fitRangeHi,sigRejectRangeLo,sigRejectRangeHi,sigIntegrateRangeLo,sigIntegrateRangeHi,canvas_tag=curr_canvas_tag,xaxis_title=curr_title)
			if(self.save_fit_lvl>0): jzFitter.SetSaveHists(self.save_fit_lvl,self.FITTED_HIST_PLOTDIR)
			results_tmp = jzFitter.FitHist3Cases(h,this_parm_list,this_var_name)
			if(results_tmp!=None):
				for key in results_dict: results_dict[key].extend( results_tmp[key] ) # key=string, val=list. Add to results
			gROOT.Reset()
			del jzFitter,this_parm_list
			this_var_name = "HiRange15MeV_narrower"
			this_parm_list=self.GetFitInitList(self.mode,MC_parms,fix_to_MC_parms)
			fitRangeLo,fitRangeHi,sigRejectRangeLo,sigRejectRangeHi,sigIntegrateRangeLo,sigIntegrateRangeHi = self.GetFitRangesByMode()
			fitRangeHi=fitRangeHi-0.015
			jzFitter = jz2GausFit(self.CXX_DIR,rebin_factor,fitRangeLo,fitRangeHi,sigRejectRangeLo,sigRejectRangeHi,sigIntegrateRangeLo,sigIntegrateRangeHi,canvas_tag=curr_canvas_tag,xaxis_title=curr_title)
			if(self.save_fit_lvl>0): jzFitter.SetSaveHists(self.save_fit_lvl,self.FITTED_HIST_PLOTDIR)
			results_tmp = jzFitter.FitHist3Cases(h,this_parm_list,this_var_name)
			if(results_tmp!=None):
				for key in results_dict: results_dict[key].extend( results_tmp[key] ) # key=string, val=list. Add to results
			gROOT.Reset()
			del jzFitter,this_parm_list
			this_var_name = "HiRange20MeV_narrower"
			this_parm_list=self.GetFitInitList(self.mode,MC_parms,fix_to_MC_parms)
			fitRangeLo,fitRangeHi,sigRejectRangeLo,sigRejectRangeHi,sigIntegrateRangeLo,sigIntegrateRangeHi = self.GetFitRangesByMode()
			fitRangeHi=fitRangeHi-0.020
			jzFitter = jz2GausFit(self.CXX_DIR,rebin_factor,fitRangeLo,fitRangeHi,sigRejectRangeLo,sigRejectRangeHi,sigIntegrateRangeLo,sigIntegrateRangeHi,canvas_tag=curr_canvas_tag,xaxis_title=curr_title)
			if(self.save_fit_lvl>0): jzFitter.SetSaveHists(self.save_fit_lvl,self.FITTED_HIST_PLOTDIR)
			results_tmp = jzFitter.FitHist3Cases(h,this_parm_list,this_var_name)
			if(results_tmp!=None):
				for key in results_dict: results_dict[key].extend( results_tmp[key] ) # key=string, val=list. Add to results
			gROOT.Reset()
			del jzFitter,this_parm_list
			this_var_name = "HiRange25MeV_narrower"
			this_parm_list=self.GetFitInitList(self.mode,MC_parms,fix_to_MC_parms)
			fitRangeLo,fitRangeHi,sigRejectRangeLo,sigRejectRangeHi,sigIntegrateRangeLo,sigIntegrateRangeHi = self.GetFitRangesByMode()
			fitRangeHi=fitRangeHi-0.025
			jzFitter = jz2GausFit(self.CXX_DIR,rebin_factor,fitRangeLo,fitRangeHi,sigRejectRangeLo,sigRejectRangeHi,sigIntegrateRangeLo,sigIntegrateRangeHi,canvas_tag=curr_canvas_tag,xaxis_title=curr_title)
			if(self.save_fit_lvl>0): jzFitter.SetSaveHists(self.save_fit_lvl,self.FITTED_HIST_PLOTDIR)
			results_tmp = jzFitter.FitHist3Cases(h,this_parm_list,this_var_name)
			if(results_tmp!=None):
				for key in results_dict: results_dict[key].extend( results_tmp[key] ) # key=string, val=list. Add to results
			gROOT.Reset()
			del jzFitter,this_parm_list
			this_var_name = "HiRange30MeV_narrower"
			this_parm_list=self.GetFitInitList(self.mode,MC_parms,fix_to_MC_parms)
			fitRangeLo,fitRangeHi,sigRejectRangeLo,sigRejectRangeHi,sigIntegrateRangeLo,sigIntegrateRangeHi = self.GetFitRangesByMode()
			fitRangeHi=fitRangeHi-0.030
			jzFitter = jz2GausFit(self.CXX_DIR,rebin_factor,fitRangeLo,fitRangeHi,sigRejectRangeLo,sigRejectRangeHi,sigIntegrateRangeLo,sigIntegrateRangeHi,canvas_tag=curr_canvas_tag,xaxis_title=curr_title)
			if(self.save_fit_lvl>0): jzFitter.SetSaveHists(self.save_fit_lvl,self.FITTED_HIST_PLOTDIR)
			results_tmp = jzFitter.FitHist3Cases(h,this_parm_list,this_var_name)
			if(results_tmp!=None):
				for key in results_dict: results_dict[key].extend( results_tmp[key] ) # key=string, val=list. Add to results
			gROOT.Reset()
			del jzFitter,this_parm_list
			
			# Background polynomial order
			this_var_name = "2ndOrderExpo"
			this_parm_list=self.GetFitInitList(self.mode,MC_parms,fix_to_MC_parms)
			this_parm_list[9] =["bkgPol2", 0.,   -100., 100.,False]
			this_parm_list[10]=["bkgPol3", 0.,   -100., 100.,True]
			jzFitter = jz2GausFit(self.CXX_DIR,rebin_factor,canvas_tag=curr_canvas_tag,xaxis_title=curr_title,*self.GetFitRangesByMode())
			if(self.save_fit_lvl>0): jzFitter.SetSaveHists(self.save_fit_lvl,self.FITTED_HIST_PLOTDIR)
			results_tmp = jzFitter.FitHist3Cases(h,this_parm_list,this_var_name)
			if(results_tmp!=None):
				for key in results_dict: results_dict[key].extend( results_tmp[key] ) # key=string, val=list. Add to results
			gROOT.Reset()
			del jzFitter,this_parm_list
			this_var_name = "3rdOrderExpo"
			this_parm_list=self.GetFitInitList(self.mode,MC_parms,fix_to_MC_parms)
			this_parm_list[9] =["bkgPol2", 0.,   -100., 100.,True]
			this_parm_list[10]=["bkgPol3", 0.,   -100., 100.,False]
			jzFitter = jz2GausFit(self.CXX_DIR,rebin_factor,canvas_tag=curr_canvas_tag,xaxis_title=curr_title,*self.GetFitRangesByMode())
			if(self.save_fit_lvl>0): jzFitter.SetSaveHists(self.save_fit_lvl,self.FITTED_HIST_PLOTDIR)
			results_tmp = jzFitter.FitHist3Cases(h,this_parm_list,this_var_name)
			if(results_tmp!=None):
				for key in results_dict: results_dict[key].extend( results_tmp[key] ) # key=string, val=list. Add to results
			gROOT.Reset()
			del jzFitter,this_parm_list
			if(self.mode=="gg" or self.mode=="3pi0"): 
				this_var_name = "2nd3rdOrderExpo"
				this_parm_list=self.GetFitInitList(self.mode,MC_parms,fix_to_MC_parms)
				this_parm_list[9] =["bkgPol2", 0.,   -100., 100.,False]
				this_parm_list[10]=["bkgPol3", 0.,   -100., 100.,False]
				jzFitter = jz2GausFit(self.CXX_DIR,rebin_factor,canvas_tag=curr_canvas_tag,xaxis_title=curr_title,*self.GetFitRangesByMode())
				if(self.save_fit_lvl>0): jzFitter.SetSaveHists(self.save_fit_lvl,self.FITTED_HIST_PLOTDIR)
				results_tmp = jzFitter.FitHist3Cases(h,this_parm_list,this_var_name)
				if (results_tmp != None):
					for key in results_dict: results_dict[key].extend(results_tmp[key])  # key=string, val=list. Add to results
				gROOT.Reset()
			if(self.mode=="3piq"): 
				this_var_name = "1stOrderExpo"
				this_parm_list=self.GetFitInitList(self.mode,MC_parms,fix_to_MC_parms)
				this_parm_list[9] =["bkgPol2", 0.,   -100., 100.,True]
				this_parm_list[10]=["bkgPol3", 0.,   -100., 100.,True]
				jzFitter = jz2GausFit(self.CXX_DIR,rebin_factor,canvas_tag=curr_canvas_tag,xaxis_title=curr_title,*self.GetFitRangesByMode())
				if(self.save_fit_lvl>0): jzFitter.SetSaveHists(self.save_fit_lvl,self.FITTED_HIST_PLOTDIR)
				results_tmp = jzFitter.FitHist3Cases(h,this_parm_list,this_var_name)
				if (results_tmp != None):
					for key in results_dict: results_dict[key].extend(results_tmp[key])  # key=string, val=list. Add to results
				gROOT.Reset()
			
		# Create dataframe for one bin, all variations specified
		df_tmp = pd.DataFrame(results_dict,columns=self.df_column_labels)
		
		# Get average of the three cases. Sometimes case0 can be negative, in which case we'd better skip.
		df_tmp["avgSigYield"]=0.
		df_tmp["avgSigYield_StatErr"]=0.
		# # Alternate version of stat. uncert. (with a minimum of +-1)
		df_tmp["Case0_SigYieldErrBinSum"]= df_tmp["Case0_HistSumErrSigBkg"]*(df_tmp["Case0_SigYield"]/(df_tmp["Case0_SigYield"]+df_tmp["BkgYield_Case0"]))**0.5
		df_tmp["Case1_SigYieldErrBinSum"]= df_tmp["Case1_HistSumErrSigBkg"]*(df_tmp["Case1_SigYield"]/(df_tmp["Case1_SigYield"]+df_tmp["BkgYield_Case1"]))**0.5
		df_tmp["Case2_SigYieldErrBinSum"]= df_tmp["Case2_HistSumErrSigBkg"]*(df_tmp["Case2_SigYield"]/(df_tmp["Case2_SigYield"]+df_tmp["BkgYield_Case2"]))**0.5
		for case in range(0,3):            df_tmp.loc[df_tmp["Case"+str(case)+"_SigYieldErrBinSum"] < 1., "Case"+str(case)+"_SigYieldErrBinSum" ] = 1. # MIN STAT ERROR
		df_tmp["NCasesAccepted"]=0
		for case in range(0,3):
			df_tmp.loc[df_tmp["Case"+str(case)+"_SigYield"] > 0.1, "avgSigYield" ]         += df_tmp["Case"+str(case)+"_SigYield"] # AVERAGE YIELD
			df_tmp.loc[df_tmp["Case"+str(case)+"_SigYield"] > 0.1, "avgSigYield_StatErr" ] += df_tmp["Case"+str(case)+"_SigYieldErrBinSum"] # AVERAGE STAT UNCERT
			df_tmp.loc[df_tmp["Case"+str(case)+"_SigYield"] > 0.1, "NCasesAccepted" ]      += 1 
		df_tmp["avgSigYield"] /= df_tmp["NCasesAccepted"]
		df_tmp["avgSigYield_StatErr"] /= df_tmp["NCasesAccepted"]
		
		df_tmp["avgSigYield"]              = (df_tmp["Case0_SigYield"]+df_tmp["Case1_SigYield"]+df_tmp["Case2_SigYield"])/3.
		df_tmp["avgSigYield_SystErrHi"]    =  (df_tmp["avgSigYield"].max()-df_tmp["avgSigYield"])
		df_tmp["avgSigYield_SystErrLo"]    = -1*(df_tmp["avgSigYield"]-df_tmp["avgSigYield"].min())
		# Set SystErrLo/Hi to zero except for "nominal" tag, confusing/misleading otherwise
		df_tmp.loc[df_tmp["variation"] != "nominal", "avgSigYield_SystErrLo" ] = 0.
		df_tmp.loc[df_tmp["variation"] != "nominal", "avgSigYield_SystErrHi" ] = 0.
		
		return df_tmp

		
	def FitMCHistGetParms(self,h,rebin_factor=2):
	
		gROOT.Reset()
		curr_title=self.HISTTITLE_DICT[self.mode].split(";")[1]
		curr_canvas_tag = self.mode+"_"+self.run+"_"+self.tag
				
		this_var_name = "nominal"
		this_parm_list=self.GetFitInitList(self.mode)
		jzFitter = jz2GausFit(self.CXX_DIR,rebin_factor,canvas_tag=curr_canvas_tag,xaxis_title=curr_title,*self.GetFitRangesByMode())
		# if(self.save_fit_lvl>0): jzFitter.SetSaveHists(self.save_fit_lvl,self.FITTED_HIST_PLOTDIR)
		results_dict = jzFitter.FitHist3Cases(h,this_parm_list,this_var_name,isDATAHist=False)
		del jzFitter,this_parm_list
		
		gROOT.Reset()
		
		return {"MC_gausMean1":results_dict["MC_gausMean1"], "MC_gausSigma1":results_dict["MC_Sigma1"], "MC_gausMean2":results_dict["MC_gausMean2"], "MC_gausSigma2":results_dict["MC_Sigma2"], "MC_gausFrac":results_dict["MC_gausFrac"], "Case1_SigYield":results_dict["Case1_SigYield"]}
		
		
	def GetETBinWindows(self):
		# STANDARD (HIGH) ENERGY BINNING
		if("LE" not in self.run):
			self.NUM_E_BINS    = 10
			self.NUM_T_BINS    = 40
			self.ebin_lo, self.ebin_hi           = 0, 9
			self.tbin_hist_lo, self.tbin_hist_hi = 0, 39
			self.NUM_T_BINS_LT = 20
			
			NUM_T_BINS_LT,NUM_T_BINS_WA,NUM_T_BINS_BA = self.NUM_T_BINS_LT,10,10 # Low t, wide angle, back angle
			self.EBIN_DIVIDER     = np.array([6.475, 7.0, 7.525, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5])
			UCHANNEL_TMIN_DIVIDER = [9.0,10.25,11.25,12.25,13.0,14.0,14.75,15.50,16.50,17.5]
			UCHANNEL_TMAX_DIVIDER = [12.0,12.9,13.8,14.75,15.70,16.65,17.6,18.5,19.45,20.4]

			self.TBINS_LO = np.zeros((self.NUM_E_BINS,self.NUM_T_BINS))
			self.TBINS_HI = np.zeros((self.NUM_E_BINS,self.NUM_T_BINS))
			for ebin in range(self.NUM_E_BINS):
				# Low t region
				step = 2./NUM_T_BINS_LT
				for tbin in range(NUM_T_BINS_LT):
					self.TBINS_LO[ebin][tbin] =tbin*step
					self.TBINS_HI[ebin][tbin] =(tbin+1.)*step
				# Wide angle region
				step = (UCHANNEL_TMIN_DIVIDER[ebin]-2.)/NUM_T_BINS_WA
				for tbin in range(NUM_T_BINS_WA):
					tbin_offset = NUM_T_BINS_LT
					self.TBINS_LO[ebin][tbin+tbin_offset] =tbin*step + 2.
					self.TBINS_HI[ebin][tbin+tbin_offset] =(tbin+1.)*step + 2.
				# Back angle region
				step = (UCHANNEL_TMAX_DIVIDER[ebin]-UCHANNEL_TMIN_DIVIDER[ebin])/NUM_T_BINS_BA
				for tbin in range(NUM_T_BINS_BA):
					tbin_offset = NUM_T_BINS_LT+NUM_T_BINS_WA
					self.TBINS_LO[ebin][tbin+tbin_offset] =tbin*step + UCHANNEL_TMIN_DIVIDER[ebin]
					self.TBINS_HI[ebin][tbin+tbin_offset] =(tbin+1.)*step + UCHANNEL_TMIN_DIVIDER[ebin]
		
			self.UCHANNEL_TMIN_DIVIDER = UCHANNEL_TMIN_DIVIDER
			self.UCHANNEL_TMAX_DIVIDER = UCHANNEL_TMAX_DIVIDER
		
		# LOW ENERGY BINNING
		if("LE" in self.run):
			self.NUM_E_BINS    = 23
			self.NUM_T_BINS    = 24
			self.ebin_lo, self.ebin_hi           = 0, 22
			self.tbin_hist_lo, self.tbin_hist_hi = 0, 23
			self.TBINS_LO     = np.zeros((self.NUM_E_BINS,self.NUM_T_BINS))
			self.TBINS_HI     = np.zeros((self.NUM_E_BINS,self.NUM_T_BINS))
			self.EBIN_DIVIDER = np.array([(2.52+i*0.04) for i in range(0,self.NUM_E_BINS+1)]) # Roughly 2.92-5.8. Although we have data+MC beyond in low E runs, no PS flux.
			CTBIN_DIVIDER     = np.array([-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.925,0.95,1.])
			for ebin in range(self.NUM_E_BINS):
				for tbin in range(self.NUM_T_BINS):
					self.TBINS_LO[ebin][tbin] = CTBIN_DIVIDER[tbin]
					self.TBINS_HI[ebin][tbin] = CTBIN_DIVIDER[tbin+1]
		# For fitting
		# Setup for bins to fit
		self.tbin_fit_lo = 1
		self.tbin_fit_hi = self.NUM_T_BINS-1
		if(self.binning_type=="WCT"): 
			self.tbin_fit_lo = 0
		
		
		
		
		
		
		
		
		
		
		
		
	def GetFitRangesByMode(self):
		if(self.mode=="gg"):
			fitRangeLo,fitRangeHi=0.4,0.7
			sigRejectRangeLo,sigRejectRangeHi=0.5,0.6
			sigIntegrateRangeLo,sigIntegrateRangeHi=0.5,0.6
		if(self.mode=="3piq"):
			fitRangeLo,fitRangeHi=0.45,0.7
			sigRejectRangeLo,sigRejectRangeHi=0.5,0.6
			sigIntegrateRangeLo,sigIntegrateRangeHi=0.5,0.6
		if(self.mode=="3pi0"):
			fitRangeLo,fitRangeHi=0.4,0.7
			sigRejectRangeLo,sigRejectRangeHi=0.46,0.6
			sigIntegrateRangeLo,sigIntegrateRangeHi=0.46,0.6
		return fitRangeLo,fitRangeHi,sigRejectRangeLo,sigRejectRangeHi,sigIntegrateRangeLo,sigIntegrateRangeHi
		
	# Parm list of lists: each inner list is of form ["string name",init,min,max,fix=True/False]
	def GetFitInitList(self,mode,MC_parms={},fix_to_MC_parms=False):
		if(mode=="gg"):
			default_parm_list_list = [
				#Signal components
				["SigYield*BinWidth", 100., 0., 1000., False], #Yeild must be positive
				["gausMean1",  0.55, 0.535, 0.560,False],         #eta mean must be close at least
				["gausSigma1", 0.01, 0.004, 0.030,False],         #eta sigma must be reasonable
				["gausFrac",   0.5,  0.000, 1.000,False],         #Yield fraction must be between 0 and 1 to be physical
				["gausMean2",  0.55, 0.535, 0.560,False],         #eta mean must be close at least
				["gausSigma2", 0.01, 0.004, 0.030,False],         #eta sigma must be reasonable
				#Background components
				["bkgAmpl",      0.0,  -0.0001, 100., False],     # exponential yield must be (almost) positive
				["bkgOffset",    0.,   0.0, 1.0000, False],      # exponential offset should be between 0 and 1 GeV
				["bkgPol1"  ,    0.,   0.0, 1000., False],     # exponential must be rising, not falling...
				["bkgPol2"  ,    0.,   -100., 100., True],     # exponential must be rising, not falling...
				["bkgPol3"  ,    0.,   -100., 100., True],     # exponential must be rising, not falling...
			]

		if(mode=="3piq"):
			default_parm_list_list = [
				#Signal components
				["SigYield*BinWidth", 100., 0., 1000., False], #Yeild must be positive
				["gausMean1",  0.55, 0.535, 0.560,False],         #eta mean must be close at least
				["gausSigma1", 0.01, 0.004, 0.020,False],         #eta sigma must be reasonable
				["gausFrac",   0.5,  0.000, 1.000,False],         #Yield fraction must be between 0 and 1 to be physical
				["gausMean2",  0.55, 0.535, 0.560,False],         #eta mean must be close at least
				["gausSigma2", 0.01, 0.004, 0.020,False],         #eta sigma must be reasonable
				#Background components
				["bkgAmpl",      0.0,  -0.0001, 100., False],     # exponential yield must be (almost) positive
				["bkgOffset",    0.,   0.0, 1.0000, False],      # exponential offset should be between 0 and 1 GeV
				["bkgPol1"  ,    0.,   0.0, 1000., False],     # exponential must be rising, not falling...
				["bkgPol2"  ,    0.,   -100., 100., False],     # exponential must be rising, not falling...
				["bkgPol3"  ,    0.,   -100., 100., False],     # exponential must be rising, not falling...
			]
		if(mode=="3pi0"):
			default_parm_list_list = [
				#Signal components
				["SigYield*BinWidth", 100., 0., 1000., False], #Yeild must be positive
				["gausMean1",  0.55, 0.525, 0.560,False],         #eta mean must be close at least
				["gausSigma1", 0.01, 0.004, 0.030,False],         #eta sigma must be reasonable
				["gausFrac",   0.5,  0.000, 1.000,False],         #Yield fraction must be between 0 and 1 to be physical
				["gausMean2",  0.55, 0.525, 0.560,False],         #eta mean must be close at least
				["gausSigma2", 0.01, 0.004, 0.030,False],         #eta sigma must be reasonable
				#Background components
				["bkgAmpl",      0.0,  -0.0001, 100., False],     # exponential yield must be (almost) positive
				["bkgOffset",    0.,   0.0, 1.0000, False],      # exponential offset should be between 0 and 1 GeV
				["bkgPol1"  ,    0.,   0.0, 1000., False],     # exponential must be rising, not falling...
				["bkgPol2"  ,    0.,   -100., 100., True],     # exponential must be rising, not falling...
				["bkgPol3"  ,    0.,   -100., 100., True],     # exponential must be rising, not falling...
			]		

		# Set initial values from MC
		if("MC_gausMean1" in MC_parms and not fix_to_MC_parms): 
			# print "RESETTING from MC parms! parm list before: \n" + str(default_parm_list_list)
			default_parm_list_list[1]=["gausMean1", MC_parms["MC_gausMean1"], default_parm_list_list[1][2],default_parm_list_list[1][3],False]
			default_parm_list_list[2]=["gausSigma1", MC_parms["MC_gausSigma1"], default_parm_list_list[2][2],default_parm_list_list[2][3],False]
			default_parm_list_list[4]=["gausMean2", MC_parms["MC_gausMean2"], default_parm_list_list[4][2],default_parm_list_list[4][3],False]
			default_parm_list_list[5]=["gausSigma2", MC_parms["MC_gausSigma2"], default_parm_list_list[5][2],default_parm_list_list[5][3],False]
			default_parm_list_list[3]=["gausFrac", MC_parms["MC_gausFrac"], default_parm_list_list[3][2],default_parm_list_list[3][3],False]
			# print "DONE RESETTING from MC parms! parm list after: \n" + str(default_parm_list_list)
		
		# Fix to MC params, if relevant
		if("MC_gausMean1" in MC_parms and fix_to_MC_parms): 
			default_parm_list_list[1]=["gausMean1", MC_parms["MC_gausMean1"], MC_parms["MC_gausMean1"], MC_parms["MC_gausMean1"],True]
			default_parm_list_list[2]=["gausSigma1", MC_parms["MC_gausSigma1"], MC_parms["MC_gausSigma1"], MC_parms["MC_gausSigma1"],True]
			default_parm_list_list[4]=["gausMean2", MC_parms["MC_gausMean2"], MC_parms["MC_gausMean2"], MC_parms["MC_gausMean2"],True]
			default_parm_list_list[5]=["gausSigma2", MC_parms["MC_gausSigma2"], MC_parms["MC_gausSigma2"], MC_parms["MC_gausMean1"],True]
			default_parm_list_list[3]=["gausFrac", MC_parms["MC_gausFrac"], MC_parms["MC_gausFrac"], MC_parms["MC_gausFrac"],True]
			
		return default_parm_list_list
			
	def AddPlottingStuff(self):
		# Plotting stuff
		self.RUN_LABELS  = {"SP17":"Spring 2017","SP18":"Spring 2018","FA18":"Fall 2018","FA18LE":"Fall 2018 Low E","SP20":"Spring 2020","COMBINED":"Combined Results"}
		self.MODE_LABELS = {"gg":"#gamma#gamma","3pi0":"#pi^{0}#pi^{0}#pi^{0}","3piq":"#pi^{+}#pi^{-}#pi^{0}","3piq_DANIEL":"#pi^{+}#pi^{-}#pi^{0} (Daniel, old)","3piq_DANIELNEW":"#pi^{+}#pi^{-}#pi^{0} (Daniel, new)","omega_gpi0":"#gamma#pi^{0}"}
		self.HISTTITLE_DICT = {
			"gg":"#eta candidates;#gamma#gamma inv. mass (GeV); Counts",
			"3piq":"#eta candidates;#pi^{+}#pi^{-}#pi^{0} inv. mass (GeV); Counts",
			"3pi0":"#eta candidates;#pi^{0}#pi^{0}#pi^{0} inv. mass (GeV); Counts",
			"omega_gpi0":"#omega candidates;#gamma#pi^{0} inv. mass (GeV); Counts",
		}
		
	def GetFluxShellExecString(self,run): 
		if(self.run!="FA18LE"): return "./jzGetFluxIndividualRuns "+str(run)+" "+str(run)+" "+str(self.RESTver)+" 1200 6. 12.0 "+str(self.FLUX_HISTS_LOC+"/"+self.run)+" "+str(self.TARGET_Z_MAX-self.TARGET_Z_MIN)
		if(self.run=="FA18LE"): return "./jzGetFluxIndividualRuns "+str(run)+" "+str(run)+" "+str(self.RESTver)+" 1200 2. 8.0  "+str(self.FLUX_HISTS_LOC+"/"+self.run)+" "+str(self.TARGET_Z_MAX-self.TARGET_Z_MIN)

	# eta-specific version of function
	def ProcessThrownHists(self, ForceRemakeThrownHists):
	
		print("Checking thrown files...")
	
		# Skip if already set (and corresponds to current run/mode)
		if(self.thrown_hists_mc!="" and os.path.exists(self.thrown_hists_mc) and self.run in self.thrown_hists_mc and self.mode in self.thrown_hists_mc): 
			print("Using pre-defined thrown file instead of default path")
			return
	
		# NOTE!!! Be sure to run `jcache put` at some point so they'll appear in /mss/
		#e.g. find /mss/halld/gluex_simulations/jzarling/eta/Aug22Gen/????/*/thrown/merged/ -name "tree_thrown*.root" -exec jcache put {} +
		# To retrieve from tape
		# find /mss/halld/gluex_simulations/jzarling/eta/Aug22Gen/????/*/thrown/merged/ -name "tree_thrown*.root" -exec jcache get {} -D 21 +
		# find /mss/halld/gluex_simulations/jzarling/eta/Aug22Gen/FA18LE/*/thrown/merged/ -name "tree_thrown*.root" -exec jcache get {} +
		
		# Folder setup
		init_dir = os.getcwd()
		thrownFileFolder = self.THROWN_ROOT_HISTS_LOC+"/thrown_"+self.run+"RESTver"+str(self.RESTver)+"_eta_"+self.mode
		if(not os.path.exists(thrownFileFolder)): os.mkdir(thrownFileFolder)
		os.chdir(thrownFileFolder)
		runs_found = 0
		# Check if thrown file already exists
		for thrown_histfile in sorted(glob.glob("./*.root")):
			run = thrown_histfile.split("_"+self.mode+"_")[1][0:5]
			if(int(run) in self.runs_to_use): runs_found+=1
		
		
		# Create from /cache/ location if needed... must be pinned!!!
		if( len(self.runs_to_use)!=runs_found ): print("Found " + str(runs_found) + " thrown histogram files, expected " + str(len(self.runs_to_use)) + "... trying to regenerate from /cache/ location...")
		if( len(self.runs_to_use)!=runs_found or ForceRemakeThrownHists): 
			thrown_tree_files = sorted(glob.glob(self.THROWN_MC_SEARCH_STRING)) # Search /cache/ location
			print("Search string: " + self.THROWN_MC_SEARCH_STRING)
			if(len(thrown_tree_files)==0):
				print("ERROR: thrown ROOT files not found... do you need to re-pin them?")
				print("Thrown ROOT tree files: " + self.THROWN_MC_SEARCH_STRING)
				sys.exit()
			hist_files = []
			for file in thrown_tree_files:
				# run_num_str = file[-11:-5] if self.mode=="gg" else file.split("_")[-3] # Filename end in "[runnum]_.root" for gg and "...[runnum]_decay_evtgen.root" for 3pi modes
				run_num_str = file[-11:-5] #if self.mode=="gg" else file.split("_")[-3] # Filename end in "[runnum]_.root" for gg and "...[runnum]_decay_evtgen.root" for 3pi modes
				outfile_name = "eta_"+self.run+"_"+self.mode+"_"+run_num_str+"_thrownHists.root"
				if(os.path.exists(outfile_name) and not ForceRemakeThrownHists): 
					if(self.VERBOSE): print("File exists, skipping run " + run_num_str)
				else: 
					exec_line = "root -b -q ../MakeThrownHists_eta.cxx+(\""+file+"\",\"eta_"+self.run+"_"+self.mode+"_"+run_num_str+"_thrownHists.root\",52.,78.,1e9,"+self.run_digit+"0000)"
					print("Executing line: \n" + exec_line)
					print("CURRENT DIRECTORY: " + os.getcwd())
					self.shell_exec(exec_line)
				hist_files.append(outfile_name)
		
		
		# hadd hists, if file did not exist (or want to overwrite)
		hadd_file_outname = "../thrownHists_"+self.run+"_"+self.mode+"_"+"RESTver"+str(self.RESTver)+"_"+str(len(self.runs_to_use))+"runs.root"
		if(not os.path.exists(hadd_file_outname) or ForceRemakeThrownHists):
			hadd_command = "hadd -f " + hadd_file_outname
			rootFilesInDir = sorted(glob.glob("eta_"+self.run+"_"+self.mode+"_*_thrownHists.root"))
			file_count=0
			for file in rootFilesInDir:
				run = int(file.split("_")[-2])
				if(run in self.runs_to_use): 
					hadd_command+=" "+file
					file_count+=1
			if(file_count<len(self.runs_to_use)):
				print("ERROR: SOME THROWN FILES ARE MISSING! WRITE SOME CODE LIKE WITH A COPIED RUNS_TO_USE THAT YOU DELETE ONCE RUN IS FOUND... or something")
				print("Number of runs: " + str(len(self.runs_to_use)))
				# print "RUNS: " + str(self.runs_to_use)
				# print "RUNS sorted: " + str(sorted(self.runs_to_use))
				print("Number of files found: " + str(file_count))
				# print "Number of thrown files found: " + str(len(rootFilesInDir))
				# return # Leave before setting location (on purpose)
				sys.exit()
			self.shell_exec(hadd_command)

		# print "Changing back directory..."
		os.chdir(init_dir)
		self.thrown_hists_mc=thrownFileFolder+"/"+hadd_file_outname



	def getThrownInETRange(self,h,elo,ehi,tlo,thi):
		if(h.GetNbinsX()!=1200):
			print("ERROR: thrown hist " + h.GetName() + " has unexpected number of bins!")
		if(h.GetNbinsY()!=2500):
			print("ERROR: thrown hist " + h.GetName() + " has unexpected number of bins!")
		xstep, ystep = 12/1200., 25/2500.
		# Fudge factor is extremely important! Bad rounding WILL happen otherwise
		xbin_lo, ybin_lo = int(floor(elo/xstep +0.0001))+1, int(floor(tlo/ystep +0.0001))+1
		xbin_hi, ybin_hi = int(floor(ehi/xstep +0.0001)),   int(floor(thi/ystep +0.0001))
		return h.Integral(xbin_lo,xbin_hi,ybin_lo,ybin_hi)

	def getThrownInWCTRange(self,h,wlo,whi,ctlo,cthi):
		if(h.GetNbinsX()!=1000):
			print("ERROR: thrown hist " + h.GetName() + " has unexpected number of bins!")
		if(h.GetNbinsY()!=1000):
			print("ERROR: thrown hist " + h.GetName() + " has unexpected number of bins!")
		xstep, ystep = 5/1000., 2/1000.
		# Fudge factor is extremely important! Bad rounding WILL happen otherwise
		xbin_lo, ybin_lo = int(floor(wlo/xstep +0.0001))+1, int(floor((ctlo+1.)/ystep +0.0001))+1
		xbin_hi, ybin_hi = int(floor(whi/xstep +0.0001)),   int(floor((cthi+1.)/ystep +0.0001))
		return h.Integral(xbin_lo,xbin_hi,ybin_lo,ybin_hi)

# ************************************************************************** #
# ********BINNING FUNCTIONS (STANDARD) ************************************* #
# ************************************************************************** #

# CALCULATE EBINS and TBINS FAST_ENOUGH FOR GLUUPY (Standard E binning)
dummy_obj    = jzEtaMesonXSec("gg","SP17","nominal")
NEBINS       = dummy_obj.NUM_E_BINS
EBIN_DIVIDER = dummy_obj.EBIN_DIVIDER
T_LO,T_HI    = dummy_obj.TBINS_LO, dummy_obj.TBINS_HI
@vectorize # Use numba vectorize (ufunc) to speed things up
def Get_ebin(E):
	for ebin in range(len(EBIN_DIVIDER)-1):
		if(EBIN_DIVIDER[ebin]<E and E<EBIN_DIVIDER[ebin+1]): return ebin # Found correct ebin
	return -1 # Else return -1
@vectorize # Use numba vectorize (ufunc) to speed things up
def Get_tbin(ebin,t):
	if(ebin<0 or ebin>=NEBINS): return -1
	for tbin in range(len(T_LO[ebin])):
		if(T_LO[ebin][tbin]<t and t<T_HI[ebin][tbin]): return tbin
	return -1 # Else return -1

# CALCULATE EBINS and TBINS FAST_ENOUGH FOR GLUUPY (LOW E)
dummy_obj_LE    = jzEtaMesonXSec("gg","FA18LE","nominal")
NEBINS_LE       = dummy_obj_LE.NUM_E_BINS
EBIN_DIVIDER_LE = dummy_obj_LE.EBIN_DIVIDER
T_LO_LE,T_HI_LE = dummy_obj_LE.TBINS_LO, dummy_obj_LE.TBINS_HI
@vectorize # Use numba vectorize (ufunc) to speed things up
def Get_ebin_LE(E):
	for ebin in range(len(EBIN_DIVIDER_LE)-1):
		if(EBIN_DIVIDER_LE[ebin]<E and E<EBIN_DIVIDER_LE[ebin+1]): return ebin # Found correct ebin
	return -1 # Else return -1
@vectorize # Use numba vectorize (ufunc) to speed things up
def Get_tbin_LE(ebin,t):
	if(ebin<0 or ebin>=NEBINS_LE): return -1
	for tbin in range(len(T_LO_LE[ebin])):
		if(T_LO_LE[ebin][tbin]<t and t<T_HI_LE[ebin][tbin]): return tbin
	return -1 # Else return -1


# ************************************************************************** #
# ********BINNING FUNCTIONS (NONSTANDARD FOR STUDIES) ********************** #
# ************************************************************************** #


# # #MODIFIED FOR SINGLE EBIN AND TBIN
# dummy_obj    = jzEtaMesonXSec("gg","SP17","nominal")
# NEBINS       = 1
# EBIN_DIVIDER = np.array([7.0, 11.5,])
# TBINS_LO = np.zeros((1,2))
# TBINS_HI = np.zeros((1,2))
# T_LO,T_HI    = np.array([[0.0,0.1,]]), np.array([[0.1,0.5,]])
# @vectorize # Use numba vectorize (ufunc) to speed things up
# def Get_ebin(E):
	# for ebin in range(len(EBIN_DIVIDER)-1):
		# if(EBIN_DIVIDER[ebin]<E and E<EBIN_DIVIDER[ebin+1]): return ebin # Found correct ebin
	# return -1 # Else return -1
# @vectorize # Use numba vectorize (ufunc) to speed things up
# def Get_tbin(ebin,t):
	# if(ebin<0 or ebin>=NEBINS): return -1
	# for tbin in range(len(T_LO[ebin])):
		# if(T_LO[ebin][tbin]<t and t<T_HI[ebin][tbin]): return tbin
	# return -1 # Else return -1

# #MODIFIED LOW E BINNING
# NEBINS_LE       = 4
# NUM_T_BINS      = 9
# EBIN_DIVIDER_LE = np.array([(2.56+i*0.22) for i in range(0,NEBINS_LE+1)]) # Roughly 2.92-5.8. Although we have data+MC beyond in low E runs, no PS flux.
# T_LO_LE = np.zeros((NEBINS_LE,NUM_T_BINS))
# T_HI_LE = np.zeros((NEBINS_LE,NUM_T_BINS))
# CTBIN_DIVIDER             = np.array([0.5,0.6,0.7,0.75,0.8,0.84,0.88,0.92,0.96,1.])
# for ebin in range(NEBINS_LE):
	# for tbin in range(NUM_T_BINS):
		# T_LO_LE[ebin][tbin] = CTBIN_DIVIDER[tbin]
		# T_HI_LE[ebin][tbin] = CTBIN_DIVIDER[tbin+1]
# @vectorize # Use numba vectorize (ufunc) to speed things up
# def Get_ebin_LE(E):
	# for ebin in range(len(EBIN_DIVIDER_LE)-1):
		# if(EBIN_DIVIDER_LE[ebin]<E and E<EBIN_DIVIDER_LE[ebin+1]): return ebin # Found correct ebin
	# return -1 # Else return -1
# @vectorize # Use numba vectorize (ufunc) to speed things up
# def Get_tbin_LE(ebin,t):
	# if(ebin<0 or ebin>=NEBINS_LE): return -1
	# for tbin in range(len(T_LO_LE[ebin])):
		# if(T_LO_LE[ebin][tbin]<t and t<T_HI_LE[ebin][tbin]): return tbin
	# return -1 # Else return -1
	