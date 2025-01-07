#!/usr/bin/env python

import sys, os.path
from optparse import OptionParser
from math import sqrt, exp, sin, cos, asin, acos, floor

#My Stuff
from jzEtaMesonXSec   import *
from jzProcessResults import *

def main(argv):
	#Usage controls from OptionParser
	parser_usage = ""
	parser = OptionParser(usage = parser_usage)
	(options, args) = parser.parse_args(argv)
	if(len(args) != 0):
		parser.print_help()
		return
	
# Just for organization: put different systematic checks here
def RunSystematicVariations(run,alt_mc_file=""):
	print("To add soon")			


def DoLowECoarserBinning(modes=["gg","3piq","3pi0"]):
	print("Add this soon too!")

def DoOtherStudiesDifferentBinning(modes=["gg","3piq","3pi0"],variations={"default":0.5,"1ns":1,"1p5ns":1.5,"2ns":2,"4ns":4,"8ns":8,"12ns":12,"20ns":20}):
	print("Did you remember to change vectorize functions???")
	print("Add soon as well")



# ////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////
# // BORING HELPER FUNCTIONS BELOW ///////////////////////////
# ////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////

# Hacky, but still easier to copy/paste than rewrite....
def InnerFCALCut(mode,cutval,cutBCAL=False,cutFCAL=False):
	cut_list=[]
	cut_list.extend([
		["g1_theta_deg","g1_theta_deg > ",cutval],
		["g2_theta_deg","g2_theta_deg > ",cutval],
	])
	if(cutBCAL):
		cut_list.extend([
			["g1_theta_deg","g1_theta_deg < ",10.8],
			["g2_theta_deg","g2_theta_deg < ",10.8],
		])
	
	if(mode=="3pi0"):
		cut_list.extend([
			["g3_theta_deg","g3_theta_deg > ",cutval],
			["g4_theta_deg","g4_theta_deg > ",cutval],
			["g5_theta_deg","g5_theta_deg > ",cutval],
			["g6_theta_deg","g6_theta_deg > ",cutval],
		])
		if(cutBCAL):
			cut_list.extend([
				["g3_theta_deg","g3_theta_deg < ",10.8],
				["g4_theta_deg","g4_theta_deg < ",10.8],
				["g5_theta_deg","g5_theta_deg < ",10.8],
				["g6_theta_deg","g6_theta_deg < ",10.8],
			])
		
	return cut_list

def OuterFCALCut(mode,cutval):
	cut_list=[]
	cut_list.extend([
		["g1_theta_deg","g1_theta_deg < ",cutval],
		["g2_theta_deg","g2_theta_deg < ",cutval],
	])
	
	if(mode=="3pi0"):
		cut_list.extend([
			["g3_theta_deg","g3_theta_deg < ",cutval],
			["g4_theta_deg","g4_theta_deg < ",cutval],
			["g5_theta_deg","g5_theta_deg < ",cutval],
			["g6_theta_deg","g6_theta_deg < ",cutval],
		])
		
	return cut_list
	
def InterfaceRegionCut(mode,cutval):
	cut_list=[]
	cut_list.extend([
		["g1_theta_deg","g1_theta_deg gt_abs ",10.8,cutval], 
		["g2_theta_deg","g2_theta_deg gt_abs ",10.8,cutval], 
	])
	
	if(mode=="3pi0"):
		cut_list.extend([
			["g2_theta_deg","g2_theta_deg gt_abs ",10.8,cutval], 
			["g2_theta_deg","g2_theta_deg gt_abs ",10.8,cutval], 
			["g2_theta_deg","g2_theta_deg gt_abs ",10.8,cutval], 
			["g2_theta_deg","g2_theta_deg gt_abs ",10.8,cutval], 
		])
		
	return cut_list
	

	
if __name__ == "__main__":
   main(sys.argv[1:])
