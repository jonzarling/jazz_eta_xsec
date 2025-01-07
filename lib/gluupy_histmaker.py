#!/usr/bin/env python3

# Standard packages
import argparse, os.path, sys, time
# These may require installation (e.g. with "pip install [package] --user" command)
import uproot
import numpy as np
from numba import jit,njit
# PyROOT: you may need to add to PYTHONPATH (e.g. setenv/export PYTHONPATH $ROOTSYS/lib:$PYTHONPATH)
from ROOT import TFile, TH1F, TH2F, TH3F, gDirectory, gROOT

# CUTS
# # Each cut on single line is a list: [branch name, cut string, value]
# # Cut is applied by parsing a cut string. It expects the format: "cut stringname [operator] ". Cut value is given by float in next list entry
# # # 1: cut quantity (this portion for human readability only, has no impact on code) 
# # # 2: arithmetic operator (examples: ">",  "<",  "==",   "!=" for greater than, less than, equals, and not equals)  
# # # If a cut is supplied that doesn't follow this form, we exit early 
ACCEPED_CUT_OPERS = ["<","<=",">",">=","==","!=","gt_abs","lt_abs"] # List of operators implemented for applying cuts.

gluupyVERBOSE = False # Either way it's modified below
GLOBAL_T0=0
GLOBAL_NUM_ENTRIES=0

np.set_printoptions(precision=4,linewidth=250)


def gluupy_setup(argv,cuts_list):
	
	# Setup parser
	parser_usage = "thisScript.py [OUTFILE.root] [INFILE1.root] [INFILE2.root ...]"
	parser = argparse.ArgumentParser(usage = parser_usage)
	# Define parser options. If no "-" or "--" flags in the add_argument below, then these are ''positional'' arguments
	parser.add_argument("-V", "-v", "--verbose",action='count',default=0, help="Print verbose output (default=off)")
	parser.add_argument("-f", dest="force_overwrite",action='count',default=0, help="Overwrite existing output file (if any)")
	parser.add_argument("-M", "--max-entries", dest="max_entries",type=float,default=-1, help="Maximum entries to process before stopping (e.g. 1000 or 1e3). If multiple input files, counting is NOT reset between them")
	parser.add_argument("--max-entries-perfile", dest="max_perfile",type=float,default=-1, help="Maximum entries to process before stopping (e.g. 1000 or 1e3). If multiple input files, counting is NOT reset between them")
	parser.add_argument("outfile",nargs='?',default="",help="output ROOT file") # The '?' consumes one command line input if possible, otherwise uses default.
	parser.add_argument("infile_list",default=[],nargs="*",help="input ROOT file(s)")
	# Now get arguments
	args = parser.parse_args(argv) # Contains all argumetns 

	# Setup bools from input flags
	global gluupyVERBOSE
	allow_overwrite= True if args.force_overwrite>=1 else False
	gluupyVERBOSE = True if args.gluupyVERBOSE>=1 else False

	# For tracking how long things take to run
	global GLOBAL_T0 
	GLOBAL_T0 = time.perf_counter()

	# Input checks
	CheckInputs(args.outfile,args.infile_list,allow_overwrite,parser_usage) # Check all intput files are .root files that exist
	CheckCutList(cuts_list) # Check that each cut in list of cuts appears valid
	max_perfile=-1 # Default value: parse ALL events
	max_perfile=int(args.max_perfile) # Convert float to int (float was desirable in options parsing so that scientific notation like 1e6 for 1,000,000 was accepted)	
	if(max_perfile!=-1): 
		print("max_perfile not implemented yet, exiting...")
		sys.exit()

	return args

def CheckInputs(outfile,infile_list,allow_overwrite=False,parser_usage=""):
	
	if(len(infile_list)==0):
		print("ERROR: no input files provided! Exiting...")
		print("Usage: \n\t" + parser_usage)
		sys.exit()
	if(len(infile_list)>=2):
		print("ERROR: No support for multiple input files yet, exiting...")
		print("INFILE LIST: " + str(infile_list))
		sys.exit()
	if(os.path.exists(outfile) and not allow_overwrite):
		print("ERROR: output file already exists! Exiting...")
		print("Usage: \n\t" + parser_usage)
		sys.exit()
	if(".root" not in outfile):
		print("ERROR: output file does not contain .root extension!\nFilename provided: "+outfile+" \nExiting...")
		print("Usage: \n\t" + parser_usage)
		sys.exit()
	
	for fname in infile_list:
		if(not os.path.exists(fname)):
			print("ERROR: input file does not exist! Exiting...")
			print("Filename: " + fname)
			sys.exit()
		if(".root" not in fname):
			print("ERROR: input file does not appear to be a ROOT file (check that filename ends in .root)")
			print("Filename: " + fname)
			sys.exit()
		
def PrintAllBranchesUproot(fname):
	ufile = uproot.open(fname)
	
	# Get keynames corresponding to TTree. There should only be one top-level unique key (if not, exit)
	all_keys=set() #Unique names only
	for key in list(ufile.keys()): all_keys.add(key.split(";")[0]) # Keynames are stores in format [key];[cycle_no]. Use split to get only the keyname
	if(len(all_keys)>=2):
		print("ERROR: more than two unique keys found in base directory of ROOT file. Check input file...")
		print("keynames found: " + str(all_keys))
		sys.exit()
	treename=""
	for key in all_keys: treename=key # Annoyingly, can't get items inside set by indexing, have to retrieve by a loop like this instead
	
	print("ALL branches: ")
	print(str(list(ufile[treename].keys())))
		
		
		
def GetBranchesUproot(fname,max_entries=-1,branchname_list=[],skipStandardBranches=False):
	
	ufile = uproot.open(fname)
	
	if(gluupyVERBOSE and max_entries!=-1): print("Retrieving only the first " + str(max_entries) + " entries in tree...")
	
	# Get keynames corresponding to TTree. There should only be one top-level unique key (if not, exit)
	all_keys=set() #Unique names only
	for key in list(ufile.keys()): all_keys.add(key.split(";")[0]) # Keynames are stores in format [key];[cycle_no]. Use split to get only the keyname
	if(len(all_keys)>=2):
		print("ERROR: more than two unique keys found in base directory of ROOT file. Check input file...")
		print("keynames found: " + str(all_keys))
		sys.exit()
	treename=""
	for key in all_keys: treename=key # Annoyingly, can't get items inside set by indexing, have to retrieve by a loop like this instead
	
	tot_file_entries=ufile[treename].num_entries
	num_entries_to_parse=max_entries if tot_file_entries>max_entries else tot_file_entries 
	
	print("Total number of entries in file  " + fname.split("/")[-1] + " " + str(tot_file_entries))
	if(max_entries!=-1): print("\tRetrieving only first " + str(max_entries) + " entries in this file ")
	
	branches_to_retrieve=[]
	if(len(branchname_list)==0): branches_to_retrieve=list(ufile[treename].keys()) # Get ALL branches
	else: branches_to_retrieve=branchname_list # Get only specified branches
	
	# Always need these three arrays to calculate FS weights, add if not already there
	if(not skipStandardBranches):
		if("event"      not in branches_to_retrieve): branches_to_retrieve.append("event")
		if("p4_beam__E" not in branches_to_retrieve): branches_to_retrieve.append("p4_beam__E")
		if("x4_beam_t"  not in branches_to_retrieve): branches_to_retrieve.append("x4_beam_t")
	
	if(gluupyVERBOSE and len(branchname_list)==0): print("Retriving ALL branches from tree...\n All branches: " + str(branches_to_retrieve))
	if(gluupyVERBOSE and len(branchname_list)!=0): print("Retriving only the following branches from tree " + str(branches_to_retrieve))
	
	br_arr_dict = {}
	# nentries = ufile[treename].num_entries
	# if(max_entries!=-1 and ufile[treename].num_entries>max_entries): nentries=max_entries
	for branch in branches_to_retrieve:
		if(gluupyVERBOSE): print("Retrieving branch: " + branch)
		np_arr = np.empty(1)
		if(max_entries==-1): np_arr = ufile[treename+"/"+branch].array(library="np")
		else:                np_arr = ufile[treename+"/"+branch].array(library="np",entry_stop=num_entries_to_parse)
		br_arr_dict[branch] = np_arr.flatten()
	
	global GLOBAL_NUM_ENTRIES
	GLOBAL_NUM_ENTRIES+=len(br_arr_dict[list(br_arr_dict.keys())[0]])
			
	return br_arr_dict

# Cut list: can be a list of lists, or a single list.
def GetEntriesPassingCuts(array_dict,cut_list):
	
	if(gluupyVERBOSE): print("Applying cuts...")
	nentries=len( array_dict[ list(array_dict.keys())[0] ] ) 
	if(gluupyVERBOSE): print("Num. entries before passing cuts " + str(nentries) + "\n")
	passes_cuts = np.ones(nentries,dtype=bool) #Initialize (all passing cuts)
	NumPassingCutsList = []
	NumSurvivedLastCut = nentries
	
	# Single cut
	if(str(type(cut_list[0]))=="<type 'str'>"): 
		cut_branchname = cut_list[0]
		cut_expression = cut_list[1]
		cut_value      = cut_list[2]
		if(cut_branchname not in list(array_dict.keys()) ):
			print("ERROR! Could not find branch " + cut_branchname + " from those available\n  All branches read in or made on-the-fly : " + str(list(array_dict.keys()))+"\n If not selecting all branches, check to see that necessary branches are included")
			sys.exit()
		if(gluupyVERBOSE): print("Applying cut: " + cut_expression + " " + str(cut_value))
		cut_op = GetWhichCutOperator(cut_expression)
		if(cut_op=="<"):      passes_cuts &= np.where(array_dict[cut_branchname]< cut_value,True,False)
		if(cut_op=="<="):     passes_cuts &= np.where(array_dict[cut_branchname]<=cut_value,True,False)
		if(cut_op==">"):      passes_cuts &= np.where(array_dict[cut_branchname]> cut_value,True,False)
		if(cut_op==">="):     passes_cuts &= np.where(array_dict[cut_branchname]>=cut_value,True,False)
		if(cut_op=="=="):     passes_cuts &= np.where(array_dict[cut_branchname]==cut_value,True,False)
		if(cut_op=="!="):     passes_cuts &= np.where(array_dict[cut_branchname]!=cut_value,True,False)
		if(cut_op=="gt_abs"): passes_cuts &= np.where( abs(array_dict[cut_branchname]-cut_value)>cut_list[i][3],True,False)
		if(cut_op=="lt_abs"): passes_cuts &= np.where( abs(array_dict[cut_branchname]-cut_value)<cut_list[i][3],True,False)
		NumPassingCuts = len(np.nonzero(passes_cuts)[0])
		if(gluupyVERBOSE):
			print("\tFraction removed by current cut " + str(NumPassingCuts/float(NumSurvivedLastCut)))
			print("\tFraction passing all cuts so far " + str(NumPassingCuts/float(nentries)) + "\n")
		NumSurvivedLastCut = NumPassingCuts

	# Multiple cuts
	elif(str(type(cut_list[0]))=="<type 'list'>"):
		for i in range(0, len(cut_list)):
			cut_branchname = cut_list[i][0]
			cut_expression = cut_list[i][1]
			cut_value      = cut_list[i][2]
			if(cut_branchname not in list(array_dict.keys()) ):
				print("ERROR! Could not find branch " + cut_branchname + " from those available\n  All branches read in or made on-the-fly : " + str(list(array_dict.keys()))+"\n If not selecting all branches, check to see that necessary branches are included")
				sys.exit()
			if(gluupyVERBOSE): print("Applying cut: " + cut_expression + " " + str(cut_value))
			cut_op = GetWhichCutOperator(cut_expression)
			if(cut_op=="<"):      passes_cuts &= np.where(array_dict[cut_branchname]< cut_value,True,False)
			if(cut_op=="<="):     passes_cuts &= np.where(array_dict[cut_branchname]<=cut_value,True,False)
			if(cut_op==">"):      passes_cuts &= np.where(array_dict[cut_branchname]> cut_value,True,False)
			if(cut_op==">="):     passes_cuts &= np.where(array_dict[cut_branchname]>=cut_value,True,False)
			if(cut_op=="=="):     passes_cuts &= np.where(array_dict[cut_branchname]==cut_value,True,False)
			if(cut_op=="!="):     passes_cuts &= np.where(array_dict[cut_branchname]!=cut_value,True,False)
			if(cut_op=="gt_abs"): passes_cuts &= np.where( abs(array_dict[cut_branchname]-cut_value)>cut_list[i][3],True,False)
			if(cut_op=="lt_abs"): passes_cuts &= np.where( abs(array_dict[cut_branchname]-cut_value)<cut_list[i][3],True,False)
			NumPassingCuts = len(np.nonzero(passes_cuts)[0])
			if(gluupyVERBOSE and nentries>1.):
				if(i!=0): print("\tFraction removed by current cut " + str(NumPassingCuts/float(NumSurvivedLastCut)))
				print("\tFraction passing all cuts so far " + str(NumPassingCuts/float(nentries)) + "\n")
			NumSurvivedLastCut = NumPassingCuts
		
	return passes_cuts
	
	
# Takes NOT of entries_passing_cuts to apply as mask (mask=True means 'bad' data in numpy parlance, sort of backwards from our usual thinking that True=passes cuts)
def ApplyCuts(branches_dict,entries_passing_cuts):
	reduced_branch_dict = dict() # initialize empty dict
	for branch in branches_dict: reduced_branch_dict[branch] =  np.ma.compressed( np.ma.array(branches_dict[branch],mask=np.logical_not(entries_passing_cuts)))
	return reduced_branch_dict

def GetFSWeightsOLDBROKEN(array_dict): 

	t0 = time.perf_counter()

	nentries = len(array_dict["event"])

	evt_diff = np.ediff1d(array_dict["event"],to_begin=-99) #Take difference of elem[i+1]-elem[i], tack a dummy value at the start. Performance seems better than sliced version of a[i]-a[i-1]
	beamE_diff = np.ediff1d(array_dict["p4_beam__E"],to_begin=-99) #Take difference of elem[i+1]-elem[i], tack a dummy value at the start. Performance seems better than sliced version of a[i]-a[i-1]
	beamT_diff = np.ediff1d(array_dict["x4_beam_t"],to_begin=-99) #Take difference of elem[i+1]-elem[i], tack a dummy value at the start. Performance seems better than sliced version of a[i]-a[i-1]
	
	# Calculate FS weight factors (if two combos accepted with same event_num and same beam photon, only difference is from FS particles, then use each with additional weight factor of 1/2 )
	# # Solution inspired by: https://stackoverflow.com/questions/42129021/counting-consecutive-1s-in-numpy-array
	three_diff = np.array([evt_diff,beamE_diff,beamT_diff])
	FSIsNew = np.abs(evt_diff) + np.abs(beamE_diff) + np.abs(beamT_diff) # 0=not a new event+beam combo. Greater than 0 = new event+beam.
	FSIsNew = np.where(FSIsNew>0.000001,1.,0.) #Make either 1 or 0
	# FSIsNew_NonZeroIndices contains all the indexes of combos that are the first encounter of that particular event+beam photon
	FSIsNew_NonZeroIndices = np.nonzero(FSIsNew)[0] # Why the [0] at the end? np.nonzero returns tuple of np_arrays. FSIsNew has 1 dimension. [1] would be second dimension, etc.
	FSIsNew_NonZeroIndices_diff = np.ediff1d(FSIsNew_NonZeroIndices,to_end=int(nentries-FSIsNew_NonZeroIndices[-1]) ) #Have to tack on the last unique event+beam photon by hand

	FS_weights=np.ones(nentries)
	counter = 0
	for val in FSIsNew_NonZeroIndices_diff:
		if(val==1): 
			counter+=1
			continue
		else:
			for i in range(val): 
				FS_weights[counter]=1./val
				counter+=1
	if(counter!=nentries): 
		print("ERROR, something went wrong!!!")
		print("Counter reached: " + str(counter))
		print("nentries: " + str(nentries))
		print("What I had so far: " + str(FS_weights))
		sys.exit()
	time_to_run = time.perf_counter()-t0
	print("Time to get final FS weight factors: " + str(time_to_run))

	return FS_weights

def GetFSWeights(array_dict): 

	t0 = time.perf_counter()
	beam_prod_np = np.round(array_dict["p4_beam__E"]*array_dict["x4_beam_t"],4) # Use beam E * beam time as unique identifier. Round just in case.
	nentries = len(array_dict["event"])

	# Nonzero index = first instance of new event/beam photon/time
	evt_diff   = np.ediff1d(array_dict["event"],to_begin=-99)        # new_arr[i]= elem[i]-elem[i-1], tack a dummy value at the start. Performance seems better than sliced version of a[i]-a[i-1]	
	
	# NEW SCHEME! Use evt diff (which I will assume is still ordered ok) and beamID only!
	# Find first index for new event, then count up number of times beam ID is encountered in like a dictionary or something
	
	FS_weights=np.ones(nentries)
	if(len(beam_prod_np)==0): return FS_weights
	n = len(evt_diff)
	
	FS_weights,running_sum,num_UniqueEventAndBeam = GetFSWeightsCore(n,FS_weights,evt_diff,beam_prod_np)
	
	# num_UniqueEventAndBeam = 0.
	# num_combo_counter = 1
	# beam_counter_in_event = {beam_prod_np[0]:1,} # First entry goes here, not in loop (helps simplify if statement)
	# running_sum = 0
	
	# for i in range(1,len(evt_diff)):
		
		# # If we reach the end, have to add last combo here by hand
		# if(i==len(evt_diff)-1): 
			# if(beam_prod_np[i] not in beam_counter_in_event): beam_counter_in_event[beam_prod_np[i]]=1
			# else: beam_counter_in_event[beam_prod_np[i]] += 1
			# num_UniqueEventAndBeam+=len(beam_counter_in_event)
			# for j in range(i-num_combo_counter,len(evt_diff)): 
				# FS_weights[j]=1./beam_counter_in_event[beam_prod_np[j]]
				# running_sum+=1./beam_counter_in_event[beam_prod_np[j]]
			# break
		
		# # If we reach a new event, modify weights
		# if(evt_diff[i]!=0):
			# num_UniqueEventAndBeam+=len(beam_counter_in_event)
			# if(i==1): num_UniqueEventAndBeam+=1 # Would miss this one otherwise
			
			# for j in range(i-num_combo_counter,i): 
				# FS_weights[j]=1./beam_counter_in_event[beam_prod_np[j]]
				# running_sum+=1./beam_counter_in_event[beam_prod_np[j]]
			# beam_counter_in_event.clear()
			# num_combo_counter=0
		
		# num_combo_counter += 1
		# if(beam_prod_np[i] not in beam_counter_in_event): beam_counter_in_event[beam_prod_np[i]]=1
		# else: beam_counter_in_event[beam_prod_np[i]] += 1
	
	if( abs(num_UniqueEventAndBeam-running_sum)>3. or abs(num_UniqueEventAndBeam-np.sum(FS_weights))>3 ):
		print("ERROR counting checks failed! Check that these three quantities match: ")
		print("SUM OF WEIGHTS (new): " + str((np.sum(FS_weights))))
		print("num_UniqueEventAndBeam: " + str(num_UniqueEventAndBeam))
		print("running_sum: " + str((running_sum)))
		sys.exit()
	
	time_to_run = time.perf_counter()-t0
	if(gluupyVERBOSE): print("Time to get final FS weight factors: " + str(time_to_run))
	
	# maybe this? https://stackoverflow.com/questions/67101069/efficient-way-to-repeatedly-split-large-numpy-array-and-record-middle
	# print "SHOULD LOOK INTO np.split OPTION FOR SPEEDUP!!!"
	# print "SHOULD LOOK INTO np.split OPTION FOR SPEEDUP!!!"
	# print "SHOULD LOOK INTO np.split OPTION FOR SPEEDUP!!!"
	# print "SHOULD LOOK INTO np.split OPTION FOR SPEEDUP!!!"
	
	return FS_weights

@njit(fastmath=True) #No-python mode just-in-time compiled code (via numba) for speedup
# n total entries in tree
# if evt_diff entry is nonzero, we've come across first entry in new event
# beam_prod_np is beam photon energy*time. Use as unique identifier (though beam ID would have been less clunky)
def GetFSWeightsCore(n,FS_weights,evt_diff,beam_prod_np):

	num_UniqueEventAndBeam = 0.
	num_combo_counter = 1
	beam_counter_in_event = {beam_prod_np[0]:1,} # First entry goes here, not in loop (helps simplify if statement)
	running_sum = 0
	
	for i in range(1,n):
		
		# If we reach the end, have to add last combo here by hand
		if(i==n-1): 
			if(beam_prod_np[i] not in beam_counter_in_event): beam_counter_in_event[beam_prod_np[i]]=1
			else: beam_counter_in_event[beam_prod_np[i]] += 1
			num_UniqueEventAndBeam+=len(beam_counter_in_event)
			for j in range(i-num_combo_counter,n): 
				FS_weights[j]=1./beam_counter_in_event[beam_prod_np[j]]
				running_sum+=1./beam_counter_in_event[beam_prod_np[j]]
			break
		
		# If we reach a new event, modify weights from last event
		if(evt_diff[i]!=0):
			num_UniqueEventAndBeam+=len(beam_counter_in_event)
			if(i==1): num_UniqueEventAndBeam+=1 # Would miss this one otherwise
			
			for j in range(i-num_combo_counter,i): 
				FS_weights[j]=1./beam_counter_in_event[beam_prod_np[j]]
				running_sum+=1./beam_counter_in_event[beam_prod_np[j]]
			beam_counter_in_event.clear()
			num_combo_counter=0
		
		num_combo_counter += 1
		if(beam_prod_np[i] not in beam_counter_in_event): beam_counter_in_event[beam_prod_np[i]]=1
		else: beam_counter_in_event[beam_prod_np[i]] += 1


	return FS_weights,running_sum,num_UniqueEventAndBeam


def ApplyCutsReduceArrays(branches_dict_in,cuts_list,calc_FSweights=True,branches_to_keep=[]):
	
	# Default: keep all input branches
	branches_dict = {}
	if(len(branches_to_keep)==0): branches_dict=branches_dict_in
	# If specified, only copy branches specified (and default branches needed for calc_FSweights)
	else:
		cut_branchnames = [cut[0] for cut in cuts_list]
		# Add only branches specified, along with ones needed for accidental subtraction / FS weights and branches for cuts
		for b in branches_to_keep+["p4_beam__E","x4_beam_t","event","accidweight"]+cut_branchnames:
			branches_dict[b]=branches_dict_in[b]
	
	entries_passing_cuts = GetEntriesPassingCuts(branches_dict,cuts_list) #Returns 1D numpy array of booleans (True=event passes cuts, False=event is cut)
	branches_dict = ApplyCuts(branches_dict,entries_passing_cuts) # Apply cuts in a second, separate step. Removes cut events from np arrays.
	if(calc_FSweights): branches_dict["FS_weight"] = GetFSWeights(branches_dict) #MUST be done after applying cuts, or else the factors here will be incorrect
	# if(calc_FSweights): branches_dict["FS_weight"] = GetFSWeightsOLDBROKEN(branches_dict) #MUST be done after applying cuts, or else the factors here will be incorrect
	return branches_dict

def CheckCutList(cut_list):

	print("Checking cut list")
	for cut in cut_list:
		# Every 'cut' in loop should be a list of 3 objects, where 
		# # Check length of list and type of all three values stored inside
		if(len(cut)!=3 and len(cut)!=4):
			print("ERROR: improper cut supplied in cut list. Three arguments are expected, but " + str(len(cut)) + " were found. Exiting...")
			sys.exit()
		if(str(type(cut[0]))!="<type 'str'>"):
			print("ERROR: first entry for this cut is not expected type 'str'. Type found instead: "+str(type(cut[0]))+". Check your list of cuts.  Exiting...")
			sys.exit()
		if(str(type(cut[1]))!="<type 'str'>"):
			print("ERROR: second entry for this cut is not expected type 'str'. Type found instead: "+str(type(cut[1]))+". Check your list of cuts.  Exiting...")
			sys.exit()
		cut_type_str = str(type(cut[2]))
		if("int" not in cut_type_str and "float" not in cut_type_str):
			print("ERROR: third entry for this cut does not match accepted types: 'int' and 'float'. Type found instead: "+cut_type_str+". Check your list of cuts.  Exiting...")
			sys.exit()

	# Check that every cut string is well formed (has math expression with something on each side of a math operator --- math operator must be implemented)
	for cut in cut_list:
		cut_op = GetWhichCutOperator(cut[1])
		if(cut_op=="ERROR"): 
			print("ERROR: cut string does not have any acceptable cut operator defined in list ACCEPED_CUT_OPERS.\n Cut string: " + cut[1] + "\n accepted operators " + str(ACCEPED_CUT_OPERS) + "\n Exiting... ")
			sys.exit()
	print("Done checking cuts")
		
	return

def FillHistFromBranchDict(h,branches_dict,branchname,DoAccidentalSub=True,DoFSWeighting=True):
	
	#First check that we have the right object before doing FillN
	if( "<class 'ROOT.TH1" not in str(type(h)) and "<class cppyy.gbl.TH1" not in str(type(h))):
		print("WARNING! Function FillHistFromBranchDict called, but histogram " + h.GetName() + " is not a 1D histogram. Skipping...")
		print("Histogram type found instead: " + str(type(h)))
		return
	
	n = len(branches_dict[branchname])
	if(n==0):
		print("Warning: call to fill histogram from 0 length array. Skipping...")
		print("\t Histogram name: " + str(h.GetName()))
		return
	
	if(DoAccidentalSub and "accidweight" not in list(branches_dict.keys())):
		print("ERROR: could not find accidental weight branch in branch_dict! Exiting...")
		sys.exit()
	if(DoFSWeighting and "FS_weight" not in list(branches_dict.keys())):
		print("ERROR: could not find FS weights branch in branch_dict! \nBe sure ApplyCutsReduceArrays is called before calling this function. Exiting...")
		sys.exit()
	if(not DoAccidentalSub and not DoFSWeighting): h.FillN(n,branches_dict[branchname].astype("float64"),np.ones(n))
	if(DoAccidentalSub and not DoFSWeighting):     h.FillN(n,branches_dict[branchname].astype("float64"),branches_dict["accidweight"].astype("float64"))
	if(DoAccidentalSub and DoFSWeighting):         h.FillN(n,branches_dict[branchname].astype("float64"),branches_dict["accidweight"]*branches_dict["FS_weight"])
	if(not DoAccidentalSub and DoFSWeighting):     h.FillN(n,branches_dict[branchname].astype("float64"),branches_dict["FS_weight"])

	return
	
def Fill2DHistFromBranchDict(h,branches_dict,branchname_x,branchname_y,DoAccidentalSub=True,DoFSWeighting=True):
	
	#First check that we have the right object before doing FillN
	if( "<class 'ROOT.TH2" not in str(type(h)) and "<class cppyy.gbl.TH2" not in str(type(h))):
		print("WARNING! Function Fill2DHistFromBranchDict called, but histogram " + h.GetName() + " is not a 2D histogram. Skipping...")
		print("Histogram type found instead: " + str(type(h)))
		return
	
	# print("TH2F TYPE: "+str(type(h)))
	
	n = len(branches_dict[branchname_x])
	if(n==0):
		print("Warning: call to fill histogram from 0 length array. Skipping...")
		print("\t Histogram name: " + str(h.GetName()))
		return
	
	if(DoAccidentalSub and "accidweight" not in list(branches_dict.keys())):
		print("ERROR: could not find accidental weight branch in branch_dict! Exiting...")
		sys.exit()
	if(DoFSWeighting and "FS_weight" not in list(branches_dict.keys())):
		print("ERROR: could not find FS weights branch in branch_dict! \nBe sure ApplyCutsReduceArrays is called before calling this function. Exiting...")
		sys.exit()
	#For 2D array, can use function FillN( n, x, y, weight, stride=1)
	if(DoAccidentalSub and DoFSWeighting):         h.FillN(n,branches_dict[branchname_x].astype("float64"),branches_dict[branchname_y].astype("float64"),branches_dict["accidweight"]*branches_dict["FS_weight"],1)
	if(not DoAccidentalSub and not DoFSWeighting): h.FillN(n,branches_dict[branchname_x].astype("float64"),branches_dict[branchname_y].astype("float64"),np.ones(n),1)
	if(DoAccidentalSub and not DoFSWeighting):     h.FillN(n,branches_dict[branchname_x].astype("float64"),branches_dict[branchname_y].astype("float64"),branches_dict["accidweight"].astype("float64"),1)
	if(not DoAccidentalSub and DoFSWeighting):     h.FillN(n,branches_dict[branchname_x].astype("float64"),branches_dict[branchname_y].astype("float64"),branches_dict["FS_weight"],1)
	# if(DoAccidentalSub and DoFSWeighting):         h.FillN(n,branches_dict[branchname_x],branches_dict[branchname_y],branches_dict["accidweight"]*branches_dict["FS_weight"],1)
	# if(not DoAccidentalSub and not DoFSWeighting): h.FillN(n,branches_dict[branchname_x],branches_dict[branchname_y],np.ones(n),1)
	# if(DoAccidentalSub and not DoFSWeighting):     h.FillN(n,branches_dict[branchname_x],branches_dict[branchname_y],branches_dict["accidweight"].astype("float64"),1)
	# if(not DoAccidentalSub and DoFSWeighting):     h.FillN(n,branches_dict[branchname_x],branches_dict[branchname_y],branches_dict["FS_weight"],1)

	return
	
# This one is more complicated... have to create a bunch of TH2 objects, fill them, then stitch them together...
def Fill3DHistFromBranchDict(h,branches_dict,branchname_x,branchname_y,branchname_z,DoAccidentalSub=True,DoFSWeighting=True):


	t0 = time.perf_counter()

	#First check that we have the right object before doing FillN
	if( "<class 'ROOT.TH3" not in str(type(h)) and "<class cppyy.gbl.TH3" not in str(type(h))):
		print("WARNING! Function Fill2DHistFromBranchDict called, but histogram " + h.GetName() + " is not a 2D histogram. Skipping...")
		print("Histogram type found instead: " + str(type(h)))
		return
	
	n = len(branches_dict[branchname_x])
	if(n==0):
		print("Warning: call to fill histogram from 0 length array. Skipping...")
		print("\t Histogram name: " + str(h.GetName()))
		return
	
	if(DoAccidentalSub and "accidweight" not in list(branches_dict.keys())):
		print("ERROR: could not find accidental weight branch in branch_dict! Exiting...")
		sys.exit()
	if(DoFSWeighting and "FS_weight" not in list(branches_dict.keys())):
		print("ERROR: could not find FS weights branch in branch_dict! \nBe sure ApplyCutsReduceArrays is called before calling this function. Exiting...")
		sys.exit()
	
	nbinsx = h.GetNbinsX()
	x_lo,x_hi = h.GetXaxis().GetBinLowEdge(1), h.GetXaxis().GetBinLowEdge(nbinsx+1)
	nbinsy = h.GetNbinsY()
	y_lo,y_hi = h.GetYaxis().GetBinLowEdge(1), h.GetYaxis().GetBinLowEdge(nbinsy+1)	
	nbinsz = h.GetNbinsY()
	z_lo, z_hi = h.GetZaxis().GetBinLowEdge(1), h.GetZaxis().GetBinLowEdge(2)
	z_step     = z_hi-z_lo
	
	for zbin in range(1,h.GetNbinsZ()+1):
		h2_tmp = TH2F("h2_tmp"+str(zbin)+"_"+h.GetName(),"", nbinsx,x_lo,x_hi, nbinsy,y_lo,y_hi )
		# CHECK LOGIC HERE!!!
		# sys.exit()
		
		cuts=[ [branchname_z,branchname_z+ " >= ",z_lo],[branchname_z,branchname_z+ " < ",z_hi] ]
		branches_dict_inBinRange = ApplyCutsReduceArrays(branches_dict,cuts,branches_to_keep=[branchname_x,branchname_y,branchname_z,"accidweight","FS_weight"])
		
		# print("cuts: " + str(cuts))
		# print("len: "  + str(len(branches_dict_inBinRange[branchname_x])))
		# print("x: "  + str(branches_dict_inBinRange[branchname_x]))
		# print("y: "  + str(branches_dict_inBinRange[branchname_y]))
		# print("z: "  + str(branches_dict_inBinRange[branchname_z]))
		# print("objects right now: ")
		# gROOT.ls()
		
		n = len(branches_dict_inBinRange[branchname_x])
		if(n==0):
			# print "Warning: call to fill histogram from 0 length array. Skipping..."
			# print "\t Histogram name: " + str(h.GetName())
			return
		
		
		#For 2D array, can use function FillN( n, x, y, weight, stride=1)
		if(DoAccidentalSub and DoFSWeighting):         h2_tmp.FillN(n,branches_dict_inBinRange[branchname_x].astype("float64"),branches_dict_inBinRange[branchname_y].astype("float64"),branches_dict_inBinRange["accidweight"]*branches_dict_inBinRange["FS_weight"],1)
		if(not DoAccidentalSub and not DoFSWeighting): h2_tmp.FillN(n,branches_dict_inBinRange[branchname_x].astype("float64"),branches_dict_inBinRange[branchname_y].astype("float64"),np.ones(n),1)
		if(DoAccidentalSub and not DoFSWeighting):     h2_tmp.FillN(n,branches_dict_inBinRange[branchname_x].astype("float64"),branches_dict_inBinRange[branchname_y].astype("float64"),branches_dict_inBinRange["accidweight"].astype("float64"),1)
		if(not DoAccidentalSub and DoFSWeighting):     h2_tmp.FillN(n,branches_dict_inBinRange[branchname_x].astype("float64"),branches_dict_inBinRange[branchname_y].astype("float64"),branches_dict_inBinRange["FS_weight"],1)
		
		# print("Mean x: "  + str(h2_tmp.GetMean(1)))
		# print("Entries: " + str(h2_tmp.GetEntries()))
		
		for xbin in range(1,nbinsx+1):
			for ybin in range(1,nbinsy+1):
				h.SetBinContent(xbin,ybin,zbin, h2_tmp.GetBinContent(xbin,ybin) )
				
		z_lo+=z_step
		z_hi+=z_step
				
		del h2_tmp,branches_dict_inBinRange

	print("time to fill 3D histogram: " + str(time.perf_counter()-t0))

	# print("objects after filling: ")
	# gROOT.ls()
		
	return h

def GetWhichCutOperator(cut_string):
	for op in ACCEPED_CUT_OPERS:
		# the < and > symbols are substrings of <= and >= respectively, consider separately
		if(op=="<" and op in cut_string and "<=" not in cut_string): return "<"
		elif(op==">" and op in cut_string and ">=" not in cut_string): return ">"
		elif(op in cut_string): return op
	# print "NOTHING FOUND FOR OP: " + op
	return "ERROR"

def SaveAllHists(fname,hist_list=[],delete_after_saving=True,open_opt="RECREATE",subdirname="",clearSubDir=False):
	
	# print("objects before saving...: ")
	# gROOT.ls()
	
	if(gluupyVERBOSE): print("Opening/creating filename: " + str(fname))
	f = TFile.Open(fname,open_opt)
	f.cd() # Need to cd before checking directories, otherwise f will be None
	# Make directory if it doesn't exist
	if(f.GetDirectory(subdirname)==None): 
		# print "Creating directory...."
		f.mkdir(subdirname)
	# Delete directory if it does exist
	elif clearSubDir:
		# print "Cleaning subdir..."
		curr_TDir = f.GetDirectory(subdirname)
		f.cd()
		f.rmdir(subdirname)
		f.mkdir(subdirname)
	f.cd()
	f.cd(subdirname)
	
	# If histograms to be saved were not specified, save ALL TH1F and TH2F objects currently open
	if(len(hist_list)==0):
	
		print("objects before saving2222...: ")
		gROOT.ls()
	
		all_objects_list = gDirectory.GetList()
		print("ALL OBJECTS ARE: " + str(len(all_objects_list)))
		for obj in all_objects_list:
			print("JZ JZ object: " + str(obj.GetName()))
			if("<class 'ROOT.TH1" or "<class 'ROOT.TH2" or "<class 'ROOT.TH3" in str(type(obj))): obj.Write() # Save 1D and 2D histograms
			if("<class 'cppyy.gbl.TH" in str(type(obj))): obj.Write() # Save 1D and 2D histograms
	# Save histograms specified in list
	else:
		for h in hist_list: h.Write()
	f.Close()
	
	# Delete histograms, if desired
	if(delete_after_saving and len(hist_list)==0):
		for obj in all_objects_list: 
			if("<class 'ROOT.TH1" in str(type(obj)) or "<class 'ROOT.TH2" in str(type(obj)) or "<class 'ROOT.TH3" in str(type(obj))): 
				gDirectory.Delete(obj.GetName()) # Save 1D histograms			gDirectory.Delete(obj.GetName())
	if(delete_after_saving and len(hist_list)!=0): 
		for h in hist_list: del h
	
	time_taken = time.perf_counter()-GLOBAL_T0
	rate = GLOBAL_NUM_ENTRIES/time_taken/1000. # kHz
	
	print("Processed " + str(GLOBAL_NUM_ENTRIES) + " entries in " + str(time_taken) + " seconds")
	print("\tor, total processing rate of " + str(rate) + " kHz")
	
	return

if __name__ == "__main__":
   main(sys.argv[1:])

