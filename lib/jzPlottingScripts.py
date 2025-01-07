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
from ROOT import TCanvas, TMath, TH2F, TH1F, TRandom, TGraphErrors, TPad, TGraphAsymmErrors, TGraph, TLine, TLegend, TLatex, TFitResultPtr, TPaveStats
from ROOT import gBenchmark, gDirectory, gROOT, gStyle, gPad, gSystem, SetOwnership

#Other packages
import numpy as np
import pandas as pd
from jz_pyroot_helper  import * # /u/home/jzarling/path/PythonPath on farm
from jzEtaMesonXSec   import *
from FitXSecHists_2GausSig_ExpBkg import jzHistFitter as jz2GausFit # Fitting function
import jzProcessResults

# Takes in df results, plots what you tell it to
class jzEtaXSecPlotter():

	# Initialize function 
	def __init__(self,plotDir,theory_dir,expt_dir,umin,umax):


		self.jzEtaMesonXSec = jzEtaMesonXSec("gg","SP17","nominal") # Dummy values, use gg for widest binning, rest not important
		self.plotDir=plotDir
	
		# initial setup, graphing choices
		self.AddPlottingVars()
		self.SavePNG = False
		self.SavePDF = True
		self.SaveSVG = False
		self.gr_offset_step            = 0.005
		self.gr_offset_step_REGION1    = 0.05
		self.marker_size       = 1.5
		self.marker_size_multi = 0.6
		self.UCHANNEL_TMIN_DIVIDER = umin
		self.UCHANNEL_TMAX_DIVIDER = umax
		self.NUM_E_BINS = 10
		self.NUM_W_BINS = 23
		self.THEORY_DIR = theory_dir
		self.EXPT_DIR = expt_dir

		# Values stored across separate draws
		self.gr_dict = {} # key=stringname, val=TGraphAsymmError
		self.legend_arr = [TLegend() for i in range(0,25)] # key=stringname, val=TGraphAsymmError
		self.gr_legend_dict = {} # key=stringname, val=TGraphAsymmError
		self.gr_maxY_dict = {} # key=stringname, val=TGraphAsymmError
		self.xaxis_range   = [np.array([]) for i in range(0,10)]
		self.xaxis_rangeLE = np.array([])
		# Canvas/Pads used in drawing
		self.can_multi = TCanvas() # Redefined later
		self.pad_multi = TPad()    # Redefined later
		self.pol0_vals = {} # key=stringname, val=[pol0,pol0_err,chi2/ndf] 

		# Old experimental results
		# # CLAS RESULTS
		self.DrawCLAS2009,self.DrawCLAS2020 = False, False
		self.CLAS2009_graphs,self.CLAS2020_graphs = {},{}
		# # OTHER OLD RESULTS
		self.DrawCornell    = False
		self.gr_Cornell, self.gr_Cornell_LE    = TGraphErrors(),TGraphErrors()
		self.gr_MIT_LE,  self.gr_DESY_LE       = TGraphErrors(),TGraphErrors()
		self.RegisterExptGraphs()

		# Theory Models
		self.EGAM_LOWE_WBINS = [2.97, 3.08, 3.19, 3.30, 3.42, 3.53, 3.65, 3.77, 3.89, 4.01, 4.14, 4.26, 4.39, 4.52, 4.65, 4.78, 4.92, 5.06, 5.19, 5.33, 5.48, 5.62, 5.76, ]
		self.THEORY_MODELS = ["JPAC","LAGET","ETAMAID","KROLL"] # Theoretical models with implemented predictions
		self.DrawJPAC=False
		self.DrawLAGET=False
		self.DrawETAMAID=False
		self.DrawKROLL=False
		self.JPAC_graphs,self.JPAC_graphsLE       = {}, {}
		self.LAGET_graphs,self.LAGET_graphsLE     = {}, {}
		self.ETAMAID_graphs,self.ETAMAID_graphsLE = {}, {}
		self.KROLL_graphs,self.KROLL_graphsLE     = {}, {}
		self.DrawReggeLTOnly=True # By default, don't plot JPAC + Laget beyond |t|>1 GeV. JPAC explicitly mentions they don't want to go beyond. Laget doesn't plot beyond either (but didn't see mention in text about range of validity). EtaMAID paper extends to nearly full range of cos(theta_cm) but doesn't really go beyond W=2800 (first 7 LE bins)
		self.RegisterAllTheoryGraphs()

		# Drawing options
		gStyle.SetOptFit(0) #Show fit results in panel
		gStyle.SetOptStat(0) #Don't show mean rms etc by default

		# for i in range(0,10):
		# 	print("LEN TBINS: " + str(self.xaxis_range[0].shape[0]))
		# 	# print("LEN TBINS: " + str(len(self.xaxis_range[0])))
		# sys.exit()

	def AddGraph(self,gr,gr_name,mode,run,ebin,yvar="xsec",legend=""):
		gr = jz_DressUpObject(gr,gr_name,xtitle="Momentum transfer |-t| (GeV^{2})",ytitle=gr.GetYaxis().GetTitle(),kColor=self.MARKER_COLOR_BY_RUN[mode],kMarkerStyle=self.MARKER_SHAPE_BY_RUN[run])
		# gr.SetMarkerSize(self.marker_size)

		isLowE = True if "FA18LE" in gr_name else False

		# More cosmetics, if we're plotting cross section specifically
		if(yvar=="xsec"):
			if(mode=="COMBINED" and run=="COMBINED"): self.gr_legend_dict[gr_name]="GlueX-I Prelim. Results"
			if(not isLowE):
				gr.SetTitle(self.TITLE_EBINS[ebin])
				gr.GetXaxis().SetTitle("Momentum transfer |-t| (GeV^{2})")
				gr.GetYaxis().SetTitle("nb per GeV^{2}")
			if(isLowE):
				gr.SetTitle(GetLowETitle(ebin))
				gr.GetXaxis().SetTitle("cos(#theta_{c.m.})")
				gr.GetYaxis().SetTitle("nb per sr")
				
		# Quantities to store for later
		self.gr_dict[gr_name]        = gr
		self.gr_legend_dict[gr_name] = self.MODE_LABELS[mode]+ " " +self.RUN_LABELS[run] + " " + legend
		self.gr_maxY_dict[gr_name]   = max(jz_tgraph2numpy(gr,"Y"))

		# print("Graph Yaxis title: " + gr.GetYaxis().GetTitle())

		return
			
	# # RECALL GRAPH NAMING SCHEME: [mode]_[run]_[tag]_[yvar]_[ebin]
	# REGION CASE:
	# (ignored for low E dataset)
	# # RegionCase: 0 for low t
	# # RegionCase: 1 for wide angle
	# # RegionCase: 2 for back angle
	# # RegionCase: 3 for ALL
	def PlotMultiResultsOneEbin(self,plot_name,modes,runs,tags,yvars=["xsec"],ebin=0,RegionCase=0,multipad=False,drawLegend=False,drawLogY=False,canvasBin=-1,FitDrawPol0=False,legend_tuple=(),graph_xmin_override=-1.,graph_xmax_override=-1.,graph_ymin_override=-1.,graph_ymax_override=-1.):

		can = TCanvas()
		
		# Draw single plot only if no pad provided
		if(not multipad): can = TCanvas("can","can",1200,900)
		
		# Draw a 5x2 plot
		if(multipad): 
			can=self.can_multi
			if(canvasBin==-1): 
				self.pad_multi.cd(ebin+1)
			if(canvasBin!=-1): 
				self.pad_multi.cd(canvasBin+1)
			gPad.SetMargin(0.05,0.01,0.05,0.01) # left,right,bottom,top
		
		drawTheoryAny = True if(self.DrawLAGET or self.DrawJPAC or self.DrawETAMAID or self.DrawKROLL) else False

		ET_binning = True if "FA18LE" not in runs else False
		
		graphs, maxY_graphs  = [],[]
		for yvar in yvars:
			for mode in modes:
				for run in runs:
					for tag in tags:
						gr_name = mode+"_"+run+"_"+tag+"_"+yvar+"_"+str(ebin)
						if(gr_name in self.gr_dict):
							graphs.append(self.gr_dict[gr_name])
							maxY_graphs.append(self.gr_maxY_dict[gr_name])
		if(multipad):
			for gr in graphs: gr.SetMarkerSize(self.marker_size_multi)

		if(len(graphs)==0):
			print("Warning, no graphs found to generate plot "+plot_name+", skipping...")
			return
			# sys.exit()
		
		# Legend for drawing
		multipleGraphs = True if(len(graphs)>2) else False
		l = TLegend( *GetLegendSizeByCase(RegionCase,ET_binning,multipleGraphs) ) if len(legend_tuple)==0 else TLegend(*legend_tuple)
		if("xsec" in yvars and ET_binning):
			if(self.DrawLAGET   and (RegionCase==0 or RegionCase==3)): l.AddEntry(self.LAGET_graphs[ebin],"J.M. Laget","pl")
			if(self.DrawJPAC    and (RegionCase==0 or RegionCase==3)): l.AddEntry(self.JPAC_graphs[ebin],"JPAC","pl")
			if(self.DrawETAMAID):                                      l.AddEntry(self.ETAMAID_graphs[ebin],"EtaMAID 2018","pl")
			if(self.DrawKROLL   and (RegionCase==1 or RegionCase==3) and ebin in self.KROLL_graphs): l.AddEntry(self.KROLL_graphs[ebin][3],"Handbag (Kroll & Passek-Kumerick)","pl")
			if(self.DrawCornell and (ebin==2 or ebin==3) and self.gr_Cornell!=None): l.AddEntry(self.gr_Cornell,"Cornell 1971 (8 GeV)","pl")
		if("xsec" in yvars and not ET_binning):
			if(self.DrawLAGET   and (RegionCase==0 or RegionCase==3) and ebin in self.LAGET_graphsLE): l.AddEntry(self.LAGET_graphsLE[ebin],"J.M. Laget","pl")
			if(self.DrawJPAC    and (RegionCase==0 or RegionCase==3) and ebin in self.JPAC_graphsLE): l.AddEntry(self.JPAC_graphsLE[ebin],"JPAC","pl")
			if(self.DrawETAMAID and ebin in self.ETAMAID_graphsLE):                                   l.AddEntry(self.ETAMAID_graphsLE[ebin],"ETAMAID","pl")
			if(self.DrawKROLL   and (RegionCase==1 or RegionCase==3) and ebin in self.KROLL_graphsLE):  l.AddEntry(self.KROLL_graphsLE[ebin][3],"Handbag (Kroll & Passek-Kumerick)","pl")
		if(not multipad): l.SetBorderSize(0) # No border around legend for big plots
		
		# Get graph ranges, and Y-axis max
		min_graph_X,max_graph_X = 0.,0.
		min_graph_Y,max_graph_Y = 0.,0.
		if(RegionCase==0):  min_graph_X,max_graph_X = 0., 2.
		if(RegionCase==1):  min_graph_X,max_graph_X = 2., self.UCHANNEL_TMIN_DIVIDER[ebin]
		if(RegionCase==2):  min_graph_X,max_graph_X = self.UCHANNEL_TMIN_DIVIDER[ebin], self.UCHANNEL_TMAX_DIVIDER[ebin]
		if(RegionCase==3):  min_graph_X,max_graph_X = 0., self.UCHANNEL_TMAX_DIVIDER[ebin]
		if(not ET_binning): min_graph_X,max_graph_X = -1.,1.
		if(graph_xmin_override>0. and graph_xmax_override>graph_xmin_override): min_graph_X,max_graph_X = graph_xmin_override,graph_xmax_override
		# Store all Y-axis points in |t| range (to get maximum *in range*)
		points_in_range=[] # Every y-axis point that will get plotted (theory or experiment)
		for gr in graphs:
			x_arr = jz_tgraph2numpy(gr,"X")
			y_arr = jz_tgraph2numpy(gr,"Y")+jz_tgraph2numpy(gr,"EYhi")
			for i,val in enumerate(x_arr):
				if(min_graph_X<val and val<max_graph_X): points_in_range.append(y_arr[i])
		# More ranges for xsec, add max of theory models, if relevant
		if("xsec" in yvars):
			if(ET_binning):
				if(RegionCase==0 or RegionCase==3):
					if(self.DrawJPAC):    points_in_range.append(max(jz_tgraph2numpy(self.JPAC_graphs[ebin],"Y")))
					if(self.DrawLAGET):   points_in_range.append(max(jz_tgraph2numpy(self.LAGET_graphs[ebin],"Y")))
					if(self.DrawETAMAID): points_in_range.append(max(jz_tgraph2numpy(self.ETAMAID_graphs[ebin],"Y")))
				if(RegionCase==2): max_graph_Y*=1.2 # Make a little larger in this case
				if(RegionCase==3): min_graph_Y = 1e-3 if ebin<7 else 1e-4
			if(not ET_binning):
				if(drawLogY): min_graph_Y=1e-2
				if(self.DrawJPAC and ebin in self.JPAC_graphsLE):       points_in_range.append(max(jz_tgraph2numpy(self.JPAC_graphsLE[ebin],"Y")))
				if(self.DrawLAGET and ebin in self.LAGET_graphsLE):     points_in_range.append(max(jz_tgraph2numpy(self.LAGET_graphsLE[ebin],"Y")))
				if(self.DrawETAMAID and ebin in self.ETAMAID_graphsLE): points_in_range.append(max(jz_tgraph2numpy(self.ETAMAID_graphsLE[ebin],"Y")))
				
		max_graph_Y = max(points_in_range)
		
		# Draw graphs
		offset_counter = 0.
		mode_counter = {"gg":0, "3pi0":0, "3piq":0, "COMBINED":0, } # Used to determine graph color
		firstPlot = True
		for gr in graphs:
			# Set graph color/size
			mode,run,tag=extract_from_graphkey(gr.GetName())
			# if(mode!="COMBINED"):
			gr.SetMarkerColor(GetGraphColor(mode,mode_counter[mode]))
			gr.SetLineColor(GetGraphColor(mode,mode_counter[mode]))
			if(multipad): 
				gr.SetMarkerSize(self.marker_size_multi)
				# print "NOTE TO SELF: ADD CODE HERE TO RESIZE MARKER AGAIN FOR SINGLE PLOTTING" 
			mode_counter[mode]+=1
			# If not first plot, draw with offset
			if(not firstPlot): 
				# Add offset to graphs and draw
				if(RegionCase!=1): offset_counter+=self.gr_offset_step
				if(RegionCase==1): offset_counter+=self.gr_offset_step_REGION1
				gr = AddOffSetToGraphX(gr,offset_counter)
				gr.Draw("psame")
				# if(RegionCase==2 or RegionCase==3): print "Warning: reminder to revisit gr_offset_step for wide angle regions"
			# More cosmetics, if on the first plot
			if(firstPlot):     
				firstPlot=False
				if(multipad): 
					gr.SetTitle(";;"+gr.GetYaxis().GetTitle())
					gr.SetMarkerSize(self.marker_size_multi)
					if(not drawTheoryAny): # Labels are smaller without theory, increase size. Seem fixed, 0.35-0.4 ish in scale.
						gr.GetXaxis().SetLabelSize( 1.1*gr.GetXaxis().GetLabelSize() )
						gr.GetYaxis().SetLabelSize( 1.1*gr.GetYaxis().GetLabelSize() )
					if(not ET_binning):
						gr.GetXaxis().SetLabelSize( 0.85*gr.GetXaxis().GetLabelSize() )
						gr.GetYaxis().SetLabelSize( 0.85*gr.GetYaxis().GetLabelSize() )
				if(not multipad): 
					if(ET_binning):
						if("xsec" in yvars): gr.SetTitle(self.TITLE_EBINS[ebin])
						gr.GetXaxis().SetTitle("Momentum transfer |-t| (GeV^{2})")
						if("xsec" in yvars): gr.GetYaxis().SetTitle("d#sigma/dt  (nb/GeV^{2})")
					if(not ET_binning):
						gr.SetTitle(GetLowETitle(ebin))
						gr.GetXaxis().SetTitle("cos(#theta_{c.m.})")
						if("xsec" in yvars): gr.GetYaxis().SetTitle("d#sigma/d#Omega  (nb/sr)")
					gr.GetYaxis().SetTitleOffset(1.2)
				if(drawLogY): gPad.SetLogy()
				# gr.GetYaxis().SetRangeUser(0,150.)
				# gr.GetYaxis().SetRangeUser(min_graph_Y,1.1*max_graph_Y)
				gr.GetHistogram().SetMinimum(min_graph_Y)
				gr.GetHistogram().SetMaximum(1.1*max_graph_Y)
				if(graph_ymax_override>1.1*max_graph_Y): gr.GetHistogram().SetMaximum(graph_ymax_override)
				if(graph_ymin_override>0):  gr.GetHistogram().SetMinimum(graph_ymin_override)

				# gr.GetHistogram().SetMaximum(150)
				gr.GetXaxis().SetLimits(min_graph_X,max_graph_X)
				# gr.GetXaxis().SetRangeUser(0.5,0.95)
				# Draw graph, and label pane if multipad
				gr.Draw("AP")
				# if(multipad): print("Multipad graph with size: " + str(gr.GetMarkerSize()))
				# if(not multipad): print("Non-multipad graph with size: " + str(gr.GetMarkerSize()))
				if(multipad):
					egam_cap = TLatex()
					egam_cap.SetTextAlign(11)
					if(RegionCase!=2):
						egam_cap.SetTextSize(0.08)
						if(ET_binning):     egam_cap.DrawLatexNDC(0.45,0.9,self.TITLE_EBINS_5x2[ebin])
						if(not ET_binning): 
							egam_cap.SetTextSize(0.06)
							egam_cap.DrawLatexNDC(0.3,0.85,GetLowETitle(ebin)[-19:])
					if(RegionCase==2):
						egam_cap.SetTextSize(0.06)
						if(ET_binning):     egam_cap.DrawLatexNDC(0.11,0.9,self.TITLE_EBINS_5x2[ebin])
						if(not ET_binning): 
							egam_cap.SetTextSize(0.06)
							egam_cap.DrawLatexNDC(0.3,0.85,GetLowETitle(ebin)[-19:])
				gr.GetXaxis().SetLimits(min_graph_X,max_graph_X)
				# print "CURRENT AXIS LABEL SIZE x,y: " + str(gr.GetXaxis().GetLabelSize())+", "+str(gr.GetYaxis().GetLabelSize())

			l.AddEntry(gr,self.gr_legend_dict[gr.GetName()],"pl") # Add graph to legend
		
		# Draw theory curves, if relevant
		if("xsec" in yvars):
			if(ET_binning):
				if(self.DrawJPAC):
					self.JPAC_graphs[ebin].Draw("csame")
				if(self.DrawLAGET):   self.LAGET_graphs[ebin].Draw("csame")
				if(self.DrawETAMAID): self.ETAMAID_graphs[ebin].Draw("csame")
				if(self.DrawKROLL and (RegionCase==1 or RegionCase==3) and ebin in self.KROLL_graphs): 
					# self.KROLL_graphs[ebin].Draw("csame")
					self.KROLL_graphs[ebin][0].Draw("fsame")
					self.KROLL_graphs[ebin][1].Draw("l")
					self.KROLL_graphs[ebin][2].Draw("l")
					self.KROLL_graphs[ebin][3].Draw("csame")
				if(self.DrawCornell and (ebin==2 or ebin==3) and self.gr_Cornell!=None): self.gr_Cornell.Draw("psame")
					
			if(not ET_binning):

				if(self.DrawJPAC    and ebin in self.JPAC_graphsLE):    self.JPAC_graphsLE[ebin].Draw("csame")
				if(self.DrawLAGET   and ebin in self.LAGET_graphsLE):   self.LAGET_graphsLE[ebin].Draw("csame")
				if(self.DrawETAMAID and ebin in self.ETAMAID_graphsLE): self.ETAMAID_graphsLE[ebin].Draw("csame")
				if(self.DrawKROLL   and ebin in self.KROLL_graphsLE): 
					self.KROLL_graphsLE[ebin][0].Draw("fsame")
					self.KROLL_graphsLE[ebin][1].Draw("l")
					self.KROLL_graphsLE[ebin][2].Draw("l")
					self.KROLL_graphsLE[ebin][3].Draw("csame")
				if(self.DrawCLAS2009 and self.CLAS2009_graphs[ebin]!=None): self.CLAS2009_graphs[ebin].Draw("psame")
				if(self.DrawCLAS2020 and self.CLAS2009_graphs[ebin]!=None): self.CLAS2020_graphs[ebin].Draw("psame")

		# Save to file, assuming we plotted anything at all
		if(firstPlot): 
			print("WARNING: no plots found to make " + plot_name + " skipping...")
			print("tagname list: " + str(tagname_list))
			print("ALL GRAPHS: "   + str(list(self.gr_dict[ebin].keys())))
			return
		
		if(FitDrawPol0):
			const_fit = TF1("const_fit","pol0",min_graph_X,max_graph_X)
			const_fit.SetLineWidth(1)
			gr.Fit(const_fit,"QBR0")
			ratio         = round(const_fit.GetParameter(0),2)
			chi2_ndf = const_fit.GetChisquare()/const_fit.GetNDF()
			l.AddEntry(const_fit,"Ratio: "+str(ratio)+" #chi^{2}/NDF="+str(round(chi2_ndf,2)),"pl")
			const_fit.Draw("same")
			self.pol0_vals[graphs[0].GetName()] = [const_fit.GetParameter(0),const_fit.GetParError(0),chi2_ndf]
			

		if(drawLegend):   
			l.Draw() # Draw legend
			self.legend_arr[ebin]=l # Make sure legend is persistent!		

		if(not multipad):
			for gr in graphs: gr.SetMarkerSize(self.marker_size)

		
		if(self.SavePNG and not multipad): 
			with suppress_stdout_stderr(): can.SaveAs(self.plotDir+"/png/"+plot_name+"_ebin"+str(ebin)+".png")
			# can.SaveAs(self.plotDir+"/png/"+plot_name+"_ebin"+str(ebin)+".png")
		if(self.SavePDF and not multipad): 
			with suppress_stdout_stderr(): can.SaveAs(self.plotDir+"/pdf/"+plot_name+"_ebin"+str(ebin)+".pdf")
			# can.SaveAs(self.plotDir+"/pdf/"+plot_name+"_ebin"+str(ebin)+".pdf")
		if(self.SaveSVG and not multipad): 
			with suppress_stdout_stderr(): can.SaveAs(self.plotDir+"/svg/"+plot_name+"_ebin"+str(ebin)+".svg")
			# can.SaveAs(self.plotDir+"/svg/"+plot_name+"_ebin"+str(ebin)+".svg")

		# # Change marker size back (needed in case we plot 1x1 later)
		# if(multipad):
		# 	for gr in graphs: gr.SetMarkerSize(self.marker_size)

		if(not multipad): 
			gPad.SetLogy(0)
			# Remove offset, in case we plot again later
			# Don't modify first graph though, else axes will get reset for multipad
			for gr in graphs[1:]: self.ResetGraphOffset(gr)
			# for gr in reversed(graphs[1:]):
			# 	gr = AddOffSetToGraphX(gr,-1*offset_counter)
			# 	if(RegionCase!=1): offset_counter-=self.gr_offset_step
			# 	if(RegionCase==1): offset_counter-=self.gr_offset_step_REGION1

		# ROOT is evil and likes to change this for no freaking reason!!!!!!! So set again just to be sure
		for i in range(len(graphs)): graphs[i].GetXaxis().SetLimits(min_graph_X,max_graph_X)


		return offset_counter

	def PlotMultiResults5x2(self,plot_name,modes,runs,tags,yvars=["xsec"],RegionCase=0,drawLegend=False,drawLogY=False,graph_ymax_override=-1.,FitDrawPol0=False):

		# print("Can multi before: " + str(self.can_multi))
		# self.can_multi.Delete()
		# self.pad_multi.Delete()
		# print("Can multi after: " + str(self.can_multi))

		# Define canvas geometry
		with suppress_stdout_stderr(): self.can_multi = TCanvas("can_multi","can_multi",900,1200)
		self.pad_multi = TPad("pad_multi", "pad_multi",0.05,0.03,0.97,0.99)
		self.pad_multi.Draw()
		self.pad_multi.cd()
		self.pad_multi.Divide(2,5,1e-10,1e-10)

		# Plot each individual pane
		last_offset = 0.
		for ebin in range(0,10): last_offset=self.PlotMultiResultsOneEbin(plot_name,modes,runs,tags,yvars=yvars,ebin=ebin,RegionCase=RegionCase,multipad=True,drawLegend=drawLegend,drawLogY=drawLogY,graph_ymax_override=graph_ymax_override,FitDrawPol0=FitDrawPol0)
		# Draw shared x, y axes
		self.can_multi.cd(0)
		xaxis_text = TLatex(0.3,0.007,"Momentum transfer |-t| (GeV^{2})")
		xaxis_text.SetTextSize(0.027)
		xaxis_text.Draw()
		yaxis_text = TLatex(0.035,0.5," d#sigma/dt  (nb/GeV^{2})")
		yaxis_text.SetTextSize(0.025)
		yaxis_text.SetTextAngle(90)
		yaxis_text.Draw()

		if(self.SavePNG): 
			with suppress_stdout_stderr(): self.can_multi.SaveAs(self.plotDir+"/png/"+plot_name+"_5x2.png")
			# self.can_multi.SaveAs(self.plotDir+"/png/"+plot_name+"_5x2.png")
		if(self.SavePDF): 
			with suppress_stdout_stderr(): self.can_multi.SaveAs(self.plotDir+"/pdf/"+plot_name+"_5x2.pdf")
			# self.can_multi.SaveAs(self.plotDir+"/pdf/"+plot_name+"_5x2.pdf")
		if(self.SaveSVG): 
			with suppress_stdout_stderr(): self.can_multi.SaveAs(self.plotDir+"/svg/"+plot_name+"_5x2.svg")
			# self.can_multi.SaveAs(self.plotDir+"/svg/"+plot_name+"_5x2.svg")


		for yvar in yvars:
			for mode in modes:
				for run in runs:
					for tag in tags:
						for ebin in range(0,10):
							gr_name = mode+"_"+run+"_"+tag+"_"+yvar+"_"+str(ebin)
							if(gr_name in self.gr_dict):
								# print("Marker size before: " + str(self.gr_dict[gr_name].GetMarkerSize()))
								self.ResetGraphOffset(self.gr_dict[gr_name])
								# print("Marker size after: " + str(self.gr_dict[gr_name].GetMarkerSize()))
							# self.removeGraphOffsets(last_offset,modes,runs,tags,yvar=yvar)

		# SetOwnership(self.can_multi, True)
		# del self.can_multi, self.pad_multi


	def PlotMultiResultsLE(self,plot_name,modes,runs,tags,yvars=["xsec"],RegionCase=0,drawLegend=False,drawLogY=False):
		del self.can_multi, self.pad_multi
		with suppress_stdout_stderr(): self.can_multi = TCanvas("can_multi","can_multi",1200,900)
		self.pad_multi = TPad("pad_multi", "pad_multi",0.05,0.05,0.97,0.99)
		self.pad_multi.Draw()
		self.pad_multi.cd()
		self.pad_multi.Divide(4,3,1e-10,1e-10)

		xaxis_text = TLatex(0.475,0.015,"cos(#theta_{c.m.})")
		xaxis_text.SetTextSize(0.0325)
		yaxis_text = TLatex(0.035,0.5," d#sigma/d#Omega  (nb/sr)")
		yaxis_text.SetTextSize(0.035)
		yaxis_text.SetTextAngle(90)

		# Plot each individual pane (first set of 12)
		for ebin in range(1,13): self.PlotMultiResultsOneEbin(plot_name,modes,runs,tags,yvars=yvars,ebin=ebin,RegionCase=4,multipad=True,drawLegend=drawLegend,drawLogY=drawLogY,canvasBin=ebin-1)
		# Draw shared x, y axes
		self.can_multi.cd(0)
		xaxis_text.Draw()
		yaxis_text.Draw()
		with suppress_stdout_stderr():
			if(drawLogY): self.can_multi.SetLogy()
			if(self.SavePNG): self.can_multi.SaveAs(self.plotDir+"/png/"+plot_name+"_LowE_Multi1.png")
			if(self.SavePDF): self.can_multi.SaveAs(self.plotDir+"/pdf/"+plot_name+"_LowE_Multi1.pdf")
			if(self.SaveSVG): self.can_multi.SaveAs(self.plotDir+"/svg/"+plot_name+"_LowE_Multi1.svg")
		# self.removeGraphOffsets(last_offset,modes,runs,tags,yvar="xsec")

		# Plot each individual pane (second set of 12)
		del self.can_multi, self.pad_multi
		with suppress_stdout_stderr(): self.can_multi = TCanvas("can_multi","can_multi",1200,900)
		self.pad_multi = TPad("pad_multi", "pad_multi",0.05,0.05,0.97,0.99)
		self.pad_multi.Draw()
		self.pad_multi.cd()
		self.pad_multi.Divide(4,3,1e-10,1e-10)
		for ebin in range(13,23): self.PlotMultiResultsOneEbin(plot_name,modes,runs,tags,yvars=yvars,ebin=ebin,RegionCase=4,multipad=True,drawLegend=drawLegend,drawLogY=drawLogY,canvasBin=ebin-13)
		# Draw shared x, y axes
		# self.can_multi.cd(12)
		# self.can_multi.Draw()
		self.can_multi.cd(0)
		xaxis_text.Draw()
		yaxis_text.Draw()
		with suppress_stdout_stderr():
			if(drawLogY): self.can_multi.SetLogy()
			if(self.SavePNG): self.can_multi.SaveAs(self.plotDir+"/png/"+plot_name+"_LowE_Multi2.png")
			if(self.SavePDF): self.can_multi.SaveAs(self.plotDir+"/pdf/"+plot_name+"_LowE_Multi2.pdf")
			if(self.SaveSVG): self.can_multi.SaveAs(self.plotDir+"/svg/"+plot_name+"_LowE_Multi2.svg")

		for yvar in yvars:
			for mode in modes:
				for run in runs:
					for tag in tags:
						for ebin in range(0, 23):
							gr_name = mode+"_"+run+"_"+tag+"_"+yvar+"_"+str(ebin)
							if(gr_name in self.gr_dict):
								self.ResetGraphOffset(self.gr_dict[gr_name])

		# self.removeGraphOffsets(last_offset,modes,runs,tags,yvar="xsec")

	# Create a graph to compare to the Cornell 4 GeV data
	# # Unfortunately I didn't see where they define
	# # df_key = mode+"_"+run+"_"+tag so probably "COMBINED_FA18LE_nominal"
	def PlotGlueX_4GeV_Comparison(self,df):

		# df = self.df_dict[df_key]
		df_curr = df[df["ebin"]==9].copy() # ebin 9 corresponds to E_gam=4.01 GeV
		W = 2.9 # Corresponds to wbin9

		from jzProcessResults import jz_DF_to_numpy

		# Refer to function calc_ct_xsec_from_E_t, but work backwards
		ct_lo             = jz_DF_to_numpy(df_curr,"tbinLo")[::-1] # Reversed order also makes sense for |t|
		ct_hi             = jz_DF_to_numpy(df_curr,"tbinHi")[::-1] # Reversed order also makes sense for |t|
		dsigma_dOmega     = jz_DF_to_numpy(df_curr,"xsec_tot")[::-1] # Reversed order also makes sense for |t|
		dsigma_dOmegaErrLo  = jz_DF_to_numpy(df_curr,"totErrLo_tot")[::-1] # Reversed order also makes sense for |t|
		dsigma_dOmegaErrHi  = jz_DF_to_numpy(df_curr,"totErrLo_tot")[::-1] # Reversed order also makes sense for |t|
		ct_avg = (ct_hi+ct_lo)/2.
		delta_ct = ct_hi-ct_lo
		nbins = len(ct_avg)

		t_lo,t_hi,t_avg = np.zeros(self.NUM_W_BINS),np.zeros(self.NUM_W_BINS),np.zeros(self.NUM_W_BINS)
		dsigma_dt, dsigma_dt_Err = np.zeros(self.NUM_W_BINS),np.zeros(self.NUM_W_BINS)

		t_lo  = -1*calc_t_from_W_ct(W,ct_hi)
		t_hi  = -1*calc_t_from_W_ct(W,ct_lo) # Reversed order also makes sense for |t|
		t_avg = -1*calc_t_from_W_ct(W,ct_avg) # Reversed order also makes sense for |t|
		tbins_err = np.zeros(nbins)
		delta_t = t_hi-t_lo
		dsigma_dt      = (2*3.14159*delta_ct)*dsigma_dOmega/delta_t
		dsigma_dtErrLo = (2*3.14159*delta_ct)*dsigma_dOmegaErrLo/delta_t
		dsigma_dtErrHi = (2*3.14159*delta_ct)*dsigma_dOmegaErrHi/delta_t

		# print "t_lo: " + str(t_lo)
		# print "t_hi: " + str(t_hi)
		# print "t_avg: " + str(t_avg)
		# print "dsigma_dt: " + str(dsigma_dt)
		# print "dsigma_dtErr: " + str(dsigma_dtErr)

		gr = TGraphAsymmErrors(nbins,t_avg,dsigma_dt,tbins_err,tbins_err,dsigma_dtErrLo,dsigma_dtErrHi)
		gr = jz_DressUpObject(gr,"gr",kColor=self.MARKER_COLOR_BY_RUN["COMBINED"],kMarkerStyle=self.MARKER_SHAPE_BY_RUN["FA18LE"])
		gr.SetTitle(GetLowETitle(9))
		gr.GetXaxis().SetTitle("Momentum transfer |-t| [GeV^{2}]")
		gr.GetYaxis().SetTitle("Cross Section [nb/GeV^{2}]")
		gr.GetYaxis().SetTitleOffset(1.2)
		gr.GetYaxis().SetTitle("nb per GeV^{2}")

		l = TLegend(0.52,0.6,0.88,0.81)
		l.AddEntry(gr,"GlueX-I 4.01 GeV (this work)","pl")
		l.AddEntry(self.gr_MIT_LE,"MIT 1968 (3.4-4.6 GeV)","pl")
		l.AddEntry(self.gr_DESY_LE,"DESY 1970 (4 GeV)","pl")
		l.AddEntry(self.gr_Cornell_LE,"Cornell 1971 (4 GeV)","pl")
		l.AddEntry(self.gr_CLAS2020_t,"CLAS 2020 (4 GeV)","pl")

		can = TCanvas("can","can",1200,900)
		gr.GetXaxis().SetRangeUser(0.,2.)
		gr.GetYaxis().SetRangeUser(0.,600.)
		gr.SetMarkerSize(1.5)
		gr.Draw("AP")
		self.gr_MIT_LE.Draw("psame")
		self.gr_DESY_LE.Draw("psame")
		self.gr_Cornell_LE.Draw("psame")
		self.gr_CLAS2020_t.Draw("psame")
		l.Draw()

		can.SaveAs(self.jzEtaMesonXSec.NOTE_PLOTS_DIR+"/pdf/RAND/CORNELL_4GeV_COMPARE.pdf")
		del can

		return

	# Create a graph to compare to the Cornell 4 GeV data
	# # Unfortunately I didn't see where they define
	# # df_key = mode+"_"+run+"_"+tag so probably "COMBINED_FA18LE_nominal"
	def PlotGlueX_8GeV_Comparison(self):

		# location: self.jzEtaMesonXSec.XSEC_RESULT_DIR + "COMBINED_RESULTS_ebin0to_ebin9.csv"
		df_key = "COMBINED_COMBINED_nominal"
		df = self.df_dict[df_key]
		df_curr = df[df["ebin"]==3].copy() # ebins 2 and 3 later?
		df_curr["totErr"] = df_curr[["totErrLo", "totErrHi"]].max(axis=1)

		from jzProcessResults import jz_df_curr_to_numpy

		# Retrieve graph values from df_curr
		nbins = len(df_curr)
		tbins_np        = (jz_DF_to_numpy(df_curr,"tbinLo")+jz_DF_to_numpy(df_curr,"tbinHi"))/2.
		tbinsErr_np     = np.zeros(nbins)
		dsigma_dt     = jz_DF_to_numpy(df_curr,"xsec_ExclIncl_tot")
		dsigma_dt_Err = jz_DF_to_numpy(df_curr,yvar_Err)

		# print "t_lo: " + str(t_lo)
		# print "t_hi: " + str(t_hi)
		# print "t_avg: " + str(t_avg)
		# print "dsigma_dt: " + str(dsigma_dt)
		# print "dsigma_dtErr: " + str(dsigma_dtErr)

		gr = TGraphAsymmErrors(nbins,t_avg,dsigma_dt,tbins_err,tbins_err,dsigma_dtErrLo,dsigma_dtErrHi)
		gr = jz_DressUpObject(gr,"gr",kColor=self.MARKER_COLOR_BY_RUN["COMBINED"],kMarkerStyle=self.MARKER_SHAPE_BY_RUN["FA18LE"])
		gr.SetTitle(GetLowETitle(9))
		gr.GetXaxis().SetTitle("Momentum transfer |-t| [GeV^{2}]")
		gr.GetYaxis().SetTitle("Cross Section [nb/GeV^{2}]")
		gr.GetYaxis().SetTitleOffset(1.2)
		gr.GetYaxis().SetTitle("nb per GeV^{2}")

		l = TLegend(0.52,0.6,0.88,0.81)
		l.AddEntry(gr,"GlueX-I 4.01 GeV (this work)","pl")
		l.AddEntry(self.gr_MIT_LE,"MIT 1968 (3.4-4.6 GeV)","pl")
		l.AddEntry(self.gr_DESY_LE,"DESY 1970 (4 GeV)","pl")
		l.AddEntry(self.gr_Cornell_LE,"Cornell 1971 (4 GeV)","pl")
		l.AddEntry(self.gr_CLAS2020_t,"CLAS 2020 (4 GeV)","pl")

		can = TCanvas("can","can",1200,900)
		gr.GetXaxis().SetRangeUser(0.,2.)
		gr.GetYaxis().SetRangeUser(0.,600.)
		gr.SetMarkerSize(1.5)
		gr.Draw("AP")
		self.gr_MIT_LE.Draw("psame")
		self.gr_DESY_LE.Draw("psame")
		self.gr_Cornell_LE.Draw("psame")
		self.gr_CLAS2020_t.Draw("psame")
		l.Draw()

		can.SaveAs(self.jzEtaMesonXSec.NOTE_PLOTS_DIR+"/pdf/RAND/CORNELL_4GeV_COMPARE.pdf")
		del can

		return


	def CalcBGGENUncert(self, bggen_hists="/w/halld-scshelf2101/home/jzarling/jzXSecTools/hist_files/BGGEN_study.root",plotHists=True,tbinsToPlot=[1,15]):
	# def CalcBGGENUncert(self, bggen_hists="/w/halld-scshelf2101/home/jzarling/jzXSecTools/hist_files/BGGEN_study.root",plotHists=True,tbinsToPlot=[i for i in range(1,16)]):

		gROOT.ProcessLine(".L "+self.jzEtaMesonXSec.CXX_DIR+"JZCustomFunctions.cxx+")
		gSystem.Load(self.jzEtaMesonXSec.CXX_DIR+"JZCustomFunctions_cxx.so")

		print("Fitting BGGEN hists....")

		legend_list = ["#gamma p #rightarrow #eta X p", "#gamma p #rightarrow non-#eta p","#gamma p #rightarrow #eta p"]
		# Check file exists
		if(not os.path.exists(bggen_hists)):
			print("ERROR: could not find histogram file for bggen sample ")
			print("Filename: " + bggen_hists)
			sys.exit()
		f = TFile.Open(bggen_hists)
		for ebin in range(0,self.NUM_E_BINS):
			if(ebin==0):
				h2_eta_bggen_incl = f.Get("gg_FA18/h2_eta_kin_ebin"+str(ebin)+"_bggen_incl")
				h2_eta_bggen_bggen_excl = f.Get("gg_FA18/h2_eta_kin_ebin"+str(ebin)+"_bggen_excl")
				h2_eta_bggen_bggen_nonEta = f.Get("gg_FA18/h2_eta_kin_ebin"+str(ebin)+"_bggen_nonEta")
			else:
				h2_eta_bggen_incl.Add(f.Get("gg_FA18/h2_eta_kin_ebin"+str(ebin)+"_bggen_incl"))
				h2_eta_bggen_bggen_excl.Add(f.Get("gg_FA18/h2_eta_kin_ebin"+str(ebin)+"_bggen_excl"))
				h2_eta_bggen_bggen_nonEta.Add(f.Get("gg_FA18/h2_eta_kin_ebin"+str(ebin)+"_bggen_nonEta"))
			if(h2_eta_bggen_incl==None or h2_eta_bggen_bggen_excl==None or h2_eta_bggen_bggen_nonEta==None):
				print("ERROR: bggen or data ROOT file found but could not locate histograms! Check syntax....")
				sys.exit()
		h_eta_bggen_incl_list = [TH2F(),]
		h_eta_bggen_excl_list = [TH2F(),]
		h_eta_bggen_nonEta_list = [TH2F(),]
		ratio_arr = np.zeros(40)
		# Tbins 1-14: sum over beam energy
		for tbin in range(1,16):
			h_eta_bggen_incl_list.append(  h2_eta_bggen_incl.ProjectionX("h_eta_bggen_incl_tbin"  +str(tbin),tbin+1,tbin+1))
			h_eta_bggen_excl_list.append(  h2_eta_bggen_bggen_excl.ProjectionX("h_eta_bggen_excl_tbin"  +str(tbin),tbin+1,tbin+1))
			h_eta_bggen_nonEta_list.append(h2_eta_bggen_bggen_nonEta.ProjectionX("h_eta_bggen_nonEta_tbin"+str(tbin),tbin+1,tbin+1))
			if(h_eta_bggen_incl_list[tbin].GetNbinsX()!=1000):
				print("ERROR: unexpected number of bins in bggen hists! Expected 1000 bins, found " + str(h_eta_bggen_incl_list.GetNbinsX()) + " bins instead")
				sys.exit()
			# Tbins 15-19: summed over several t bins for better stats
			if(tbin==15):
				h_eta_bggen_incl_list.append(  h2_eta_bggen_incl.ProjectionX("h_eta_bggen_incl_tbin"  +str(tbin),16,19))
				h_eta_bggen_excl_list.append(  h2_eta_bggen_bggen_excl.ProjectionX("h_eta_bggen_excl_tbin"  +str(tbin),16,19))
				h_eta_bggen_nonEta_list.append(h2_eta_bggen_bggen_nonEta.ProjectionX("h_eta_bggen_nonEta_tbin"+str(tbin),16,19))

			# Fit to with/without incl eta part to see the difference
			savePlotLvl = 2 if plotHists else 0
			h_excl_nonEtaBkg = h_eta_bggen_excl_list[tbin].Clone()
			h_excl_nonEtaBkg.Add(h_eta_bggen_nonEta_list[tbin])
			jzFitter = jz2GausFit(self.jzEtaMesonXSec.CXX_DIR,2,canvas_tag="bggen",xaxis_title=self.HISTTITLE_DICT["gg"].split(";")[1],*self.jzEtaMesonXSec.GetFitRangesByMode())
			jzFitter.VERBOSE = False
			jzFitter.max_chi2 = 1.e16
			jzFitter.SetSaveHists(savePlotLvl, "/w/halld-scshelf2101/home/jzarling/jzXSecTools/more_plots/bggen_fits/")
			this_parm_list = self.jzEtaMesonXSec.GetFitInitList("gg")
			results_dict_excl_nonEtaBkg = jzFitter.FitHist3Cases(h_excl_nonEtaBkg, this_parm_list, "bggen_excl_nonEta")
			del jzFitter, this_parm_list

			# print("results_dict_excl_nonEtaBkg: " + str(results_dict_excl_nonEtaBkg))

			tot_hist = h_eta_bggen_excl_list[tbin].Clone()
			tot_hist.Add(h_eta_bggen_incl_list[tbin])
			tot_hist.Add(h_eta_bggen_nonEta_list[tbin])
			jzFitter = jz2GausFit(self.jzEtaMesonXSec.CXX_DIR,2,canvas_tag="bggen",xaxis_title=self.HISTTITLE_DICT["gg"].split(";")[1],*self.jzEtaMesonXSec.GetFitRangesByMode())
			jzFitter.VERBOSE = False
			jzFitter.max_chi2 = 1.e16
			jzFitter.SetSaveHists(savePlotLvl, "/w/halld-scshelf2101/home/jzarling/jzXSecTools/more_plots/bggen_fits/")
			this_parm_list = self.jzEtaMesonXSec.GetFitInitList("gg")
			results_dict_tot = jzFitter.FitHist3Cases(tot_hist, this_parm_list, "bggen_tot")
			del jzFitter, this_parm_list

			# print("results_dict_tot: " + str(results_dict_tot))


			if(results_dict_tot==None or results_dict_excl_nonEtaBkg==None):
				print("ERROR IN FITTING BGGEN HIST! Exiting...")
				sys.exit()

			num_incl   = h_eta_bggen_incl_list[tbin].Integral(451,651)
			num_excl   = h_eta_bggen_excl_list[tbin].Integral()
			num_nonEtaInRange = h_eta_bggen_nonEta_list[tbin].Integral()

			excess_list = [results_dict_tot["Case0_SigYield"][0]-results_dict_excl_nonEtaBkg["Case0_SigYield"][0],
						  results_dict_tot["Case1_SigYield"][0]-results_dict_excl_nonEtaBkg["Case1_SigYield"][0],
						  results_dict_tot["Case2_SigYield"][0]-results_dict_excl_nonEtaBkg["Case2_SigYield"][0]]
			for val in excess_list:
				if val > num_incl/num_nonEtaInRange: excess_list.remove(val)
			if(len(excess_list)==0): excess_list.append(num_incl/num_nonEtaInRange)
			excess = max(excess_list)

			ratio_arr[tbin]=excess/num_nonEtaInRange

			# If fit suggests larger inclusive count than true value, reset to true (unlikely but happens)
			if(num_incl/num_nonEtaInRange < ratio_arr[tbin]): ratio_arr[tbin]=num_incl/num_nonEtaInRange

			if(plotHists and tbin in tbinsToPlot): self.jzPlotHistStack([h_eta_bggen_incl_list[tbin], h_eta_bggen_nonEta_list[tbin], h_eta_bggen_excl_list[tbin]],legend_list,"CH4/bggen_eta_gg_tbin" + str(tbin), 4, 0., 1.,hstack_title=";#gamma#gamma inv. mass (GeV);Counts / 4 MeV")

		# sys.exit()
		# Rest of t-range: use last few |t| bins averaged
		for tbin in range(16, 40): ratio_arr[tbin]=ratio_arr[15]


		return ratio_arr

	# Both range_lo and range_hi need to be set to non-defaults in order to be used
	def jzPlotHistStack(self,h_list, legend_list, tag_name, rebin_factor=1, range_lo=-1000, range_hi=1000, legend_limits=[],reverse=False, hstack_title=""):
		ctmp = TCanvas("ctmp", "ctmp", 1200, 900)
		hstack = THStack(tag_name, "")

		legend = TLegend()
		if (len(legend_limits) == 4): legend = TLegend(legend_limits[0], legend_limits[1], legend_limits[2], legend_limits[3])
		else: legend = TLegend(0.11, 0.75, 0.4,0.89)  # 0.1 is lower limit of plot, 0.9 is upper limit (beyond on either side is labeling+whitespace)

		if (not reverse):
			for i in range(0, len(h_list)):
				if (rebin_factor != 1): h_list[i].Rebin(rebin_factor)
				h_list[i].SetLineColor(i + 1)  # Start from 1 for better kColors (0 is white)
				h_list[i].SetFillColor(i + 1)  # Start from 1 for better kColors (0 is white)
				legend.AddEntry(h_list[i], legend_list[i], "pl")
				hstack.Add(h_list[i])
		if (reverse):
			for i in reversed(range(0, len(h_list))):
				if (rebin_factor != 1): h_list[i].Rebin(rebin_factor)
				h_list[i].SetLineColor(i + 1)  # Start from 1 for better kColors (0 is white)
				h_list[i].SetFillColor(i + 1)  # Start from 1 for better kColors (0 is white)
				legend.AddEntry(h_list[i], legend_list[i], "pl")
				hstack.Add(h_list[i])

		if (range_lo != -1000 and range_hi != 1000):
			hstack.Draw("")  # Have to draw before doing setrangeuser??? Well ok then...
			hstack.GetXaxis().SetRangeUser(range_lo, range_hi)

		# Set axis titles of hstack to match first histogram
		hstack.GetXaxis().SetTitle(h_list[0].GetXaxis().GetTitle())
		hstack.GetYaxis().SetTitle(h_list[0].GetYaxis().GetTitle())
		hstack.SetTitle(hstack_title)

		hstack.Draw("hist")
		legend.Draw()
		if(self.SavePNG):
			with suppress_stdout_stderr(): ctmp.SaveAs(self.plotDir+"/png/"+tag_name+".png")
			# ctmp.SaveAs(self.plotDir+"/png/"+tag_name+".png")
		if(self.SavePDF):
			with suppress_stdout_stderr(): ctmp.SaveAs(self.plotDir+"/pdf/"+tag_name+".pdf")
			# ctmp.SaveAs(self.plotDir+"/pdf/"+tag_name+".pdf")
		if(self.SaveSVG):
			with suppress_stdout_stderr(): ctmp.SaveAs(self.plotDir+"/svg/"+tag_name+".svg")
			# ctmp.SaveAs(self.plotDir+"/svg/"+tag_name+".svg")

		del ctmp
		return hstack

	def ResetGraphOffset(self, gr):


		gr_name = gr.GetName()
		ebin    = int(gr.GetName().split("_")[-1])
		isLE = True if "FA18LE" in gr_name else False

		xaxis = self.xaxis_range[ebin] if not isLE else self.xaxis_rangeLE
		yaxis = jz_tgraph2numpy(gr, "Y")
		# print("graphname: " + gr.GetName())
		# print("ebin: " + str(ebin))
		# print("X shape: " + str(xaxis.shape))
		# print("Y shape: " + str(yaxis.shape))
		if(not isLE):
			for i in range(gr.GetN()): gr.SetPoint(i, xaxis[i], yaxis[i])

		gr.SetMarkerSize(self.marker_size)

		return gr

	def removeGraphOffsets(self,offset,modes,runs,tags,yvar="xsec",num_ebins=10):
		for ebin in range(0,num_ebins):
			offset_counter=offset
			graphs=[]
			for mode in modes:
				for run in runs:
					for tag in tags:
						if(mode+"_"+run+"_"+tag+"_"+yvar+"_"+str(ebin) not in self.gr_dict): continue
						else: graphs.append(self.gr_dict[mode+"_"+run+"_"+tag+"_"+yvar+"_"+str(ebin)])
			if(len(graphs)==0): return
			
			for gr in reversed(graphs[1:]):
				gr = AddOffSetToGraphX(gr,-1*offset_counter)
				offset_counter-=self.gr_offset_step
			drawTheoryAny = True if(self.DrawLAGET or self.DrawJPAC or self.DrawETAMAID or self.DrawKROLL) else False
			if(not drawTheoryAny): # Also change label sizes back
				graphs[0].GetXaxis().SetLabelSize( (1/1.1)*graphs[0].GetXaxis().GetLabelSize() )
				graphs[0].GetYaxis().SetLabelSize( (1/1.1)*graphs[0].GetYaxis().GetLabelSize() )
		return

	def RegisterExptGraphs(self):
		for ebin in range(0,self.NUM_W_BINS):
			gr_CLAS2020=TGraphErrors()
			gr_CLAS2009=TGraphAsymmErrors()
			if(ebin<5):  gr_CLAS2009 = self.GetCLAS2009Graph(ebin)
			if(1 < ebin and ebin<12): gr_CLAS2020 = self.GetCLAS2020Graph(ebin)
			self.CLAS2009_graphs[ebin] = gr_CLAS2009
			self.CLAS2020_graphs[ebin] = gr_CLAS2020
		jz_opacity = 0.25
		# MIT 1968 POINTS! https://www.hepdata.net/record/ins53018
		# # 4 GeV points in units of nb/GeV^2 (3.4-4.6 spread)
		x_arr_LE     = np.array([0.04, 0.08, 0.18, 0.40, 0.67, 1.00])
		x_arr_Err_LE = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
		y_arr_LE     = np.array([80.0, 285., 240., 160., 100., 175.])
		y_arr_Err_LE = np.array([50.0, 60.0, 50.0, 30.0, 40.0, 45.0])
		self.gr_MIT_LE = jz_DressUpObject(TGraphErrors(6,x_arr_LE,y_arr_LE,x_arr_Err_LE,y_arr_Err_LE),"gr_MIT_LE",kColor=kRed,kMarkerStyle=kOpenCircle)
		self.gr_MIT_LE.SetLineColorAlpha(kRed,jz_opacity)
		# DESY POINTS! https://www.hepdata.net/record/ins63065
		# # 4 GeV points in units of nb/GeV^2
		x_arr_LE     = np.array([0.017, 0.028, 0.063, 0.12, 0.21, 0.31, 0.44, 0.59, 0.76, 0.94, 1.12, 1.37])
		x_arr_Err_LE = np.array([0.000, 0.000, 0.000, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
		y_arr_LE     = np.array([295.0, 270.0, 406.0, 408., 432., 330., 234., 153., 128., 62.0, 41.0, 44.0])
		y_arr_Err_LE = np.array([50.00, 49.00, 62.00, 58.0, 65.0, 50.0, 42.0, 34.0, 30.0, 22.0, 24.0, 22.0, 00.0,])
		self.gr_DESY_LE = jz_DressUpObject(TGraphErrors(12,x_arr_LE,y_arr_LE,x_arr_Err_LE,y_arr_Err_LE),"gr_DESY_LE",kColor=kBlue,kMarkerStyle=kOpenCircle )
		self.gr_DESY_LE.SetLineColorAlpha(kBlue,jz_opacity)
		# CORNELL POINTS! https://www.hepdata.net/record/ins75928
		# # 4 GeV points in units of nb/GeV^2 (+/- 9% FWHM, at least at 5 GeV)
		x_arr_LE     = np.array([0.30, 0.40, 0.50, 0.65, 0.80])
		x_arr_Err_LE = np.array([0.00, 0.00, 0.00, 0.00, 0.00])
		y_arr_LE     = np.array([279., 273., 356., 201., 145.])
		y_arr_Err_LE = np.array([33.4, 33.0, 29.5, 23.0, 19.4])
		self.gr_Cornell_LE = jz_DressUpObject(TGraphErrors(5,x_arr_LE,y_arr_LE,x_arr_Err_LE,y_arr_Err_LE),"gr_Cornell_LE",kColor=kDarkGreen,kMarkerStyle=kOpenCircle )
		self.gr_Cornell_LE.SetLineColorAlpha(kDarkGreen,jz_opacity)
		# CORNELL POINTS!  (+/- 9% FWHM, at least at 5 GeV)
		# 8 GeV data
		x_arr     = np.array([0.30, 0.40, 0.50, 0.65, 0.80])
		x_arr_Err = np.array([0.00, 0.00, 0.00, 0.00, 0.00])
		y_arr     = np.array([69.2, 61.2, 60.8, 37.2, 25.8])
		y_arr_Err = np.array([12.6, 12.1, 12.0, 10.5, 10.0])
		self.gr_Cornell = jz_DressUpObject(TGraphErrors(5,x_arr,y_arr,x_arr_Err,y_arr_Err),"gr_Cornell",kColor=kDarkGreen,kMarkerStyle=kOpenCircle )
		self.gr_Cornell.SetLineColorAlpha(kDarkGreen,jz_opacity)
		# CLAS 2020 (over t)!
		# 8 GeV data
		x_arr     = np.array([0.500, 0.70, 0.90, 1.10, 1.30, 1.50, 1.70, 1.9, 2.1, 2.3, 2.5, 2.7, 3.1, 3.3, 3.5, 3.7, 3.9])
		y_arr     = np.array([106.2, 84.9, 49.6, 17.3, 16.3, 12.9, 22.7, 5.3, 3.5, 2.1, 3.4, 2.4, 0.0, 1.4, 2.2, 1.5, 1.0])
		x_arr_Err = np.zeros(len(x_arr))
		y_arr_Err = np.array([47.75, 32.7, 16.9, 12.7, 13.4, 8.04, 10.8, 5.5, 4.9, 3.7, 3.4, 3.0, 3.2, 3.2, 4.6, 2.3, 2.9])
		self.gr_CLAS2020_t = jz_DressUpObject(TGraphErrors(len(x_arr),x_arr,y_arr,x_arr_Err,y_arr_Err),"gr_CLAS2020_t",kColor=kMagenta,kMarkerStyle=kOpenCircle )
		self.gr_CLAS2020_t.SetLineColorAlpha(kMagenta,jz_opacity)

	def GetCLAS2009Graph(self, wbin):

		# My bin 0 is their bin 59, they have lots of even lower energy data
		offset = 59
		if (wbin + offset > 64): return

		f = TFile.Open(self.EXPT_DIR+"CLAS_2009.root")
		gr_CLAS = f.Get("Table " + str(wbin + offset) + "/Graph1D_y1")

		gr_CLAS_xaxis = jz_tgraph2numpy(gr_CLAS, "X")
		gr_CLAS_yaxis = jz_tgraph2numpy(gr_CLAS, "Y")
		gr_CLAS_yaxisErrLo = jz_tgraph2numpy(gr_CLAS, "EYlo")
		gr_CLAS_yaxisErrHi = jz_tgraph2numpy(gr_CLAS, "EYhi")

		x_axis_offset = -0.005

		for i in range(gr_CLAS.GetN()):
			gr_CLAS.SetPoint(i, gr_CLAS_xaxis[i] + x_axis_offset, gr_CLAS_yaxis[i] * 1000.)  # Convert from ub to nb
			gr_CLAS.SetPointEXhigh(i, 0.)  # No x-axis error for consistency
			gr_CLAS.SetPointEXlow(i, 0.)  # No x-axis error for consistency
			gr_CLAS.SetPointEYhigh(i, gr_CLAS_yaxisErrHi[i] * 1000.)  # Convert from ub to nb
			gr_CLAS.SetPointEYlow(i, gr_CLAS_yaxisErrLo[i] * 1000.)  # Convert from ub to nb

		gr_CLAS.SetLineColor(kBlue + 1)
		gr_CLAS.SetMarkerColor(kBlue + 1)
		gr_CLAS.SetMarkerStyle(kOpenCircle)
		gr_CLAS.SetMarkerSize(self.marker_size_multi)



		return gr_CLAS

	def GetCLAS2020Graph(self, wbin):
		txt_loc = self.EXPT_DIR+"W_1760_3120.txt"

		w_center = (LOWE_W_DIVIDER[wbin] + LOWE_W_DIVIDER[wbin + 1]) / 2.
		if (wbin >= 12): return

		x_axis_offset = 0.005

		CLAS_txt = open(txt_loc, "r")
		CLAS_ct = array('d', [])
		CLAS_sigma = array('d', [])
		CLAS_sigmaErr = array('d', [])
		for line in CLAS_txt.readlines():
			line_arr = line.split()
			if (len(line_arr) != 5 or line[0] == "W"): continue
			if (abs(w_center - float(line_arr[0])) > 0.001): continue  # Skip wrong W bin
			CLAS_ct.append(float(line_arr[1]) + x_axis_offset)
			CLAS_sigma.append(float(line_arr[2]) * 1000.)  # Convert ub to nb
			CLAS_sigmaErr.append(sqrt(float(line_arr[3]) ** 2 + float(line_arr[4]) ** 2))
		gr = TGraphErrors(len(CLAS_ct), CLAS_ct, CLAS_sigma, np.full(len(CLAS_ct), 0.1), CLAS_sigmaErr)

		gr.SetLineColor(kBlue - 7)
		gr.SetMarkerColor(kBlue - 7)
		gr.SetMarkerStyle(26)  # Open triangle
		gr.SetMarkerSize(self.marker_size_multi)

		gr_EY = jz_tgraph2numpy(gr, "EY")
		for i in range(gr.GetN()): gr.SetPointError(i, 0., gr_EY[i] * 1000.)

		gr.SetTitle("")

		return gr

	def RegisterAllTheoryGraphs(self):
		# Standard energy version
		for model in self.THEORY_MODELS:
			for ebin in range(0,self.NUM_E_BINS): self.AddTheoryGraph(model,ebin)
		# Low energy version
		for model in self.THEORY_MODELS:
			for ebin in range(1,self.NUM_W_BINS):
				self.AddTheoryGraph(model,ebin,standardE=False)

	# Add all theory graphs
	def AddTheoryGraph(self, model, ebin, standardE=True):

		if (not standardE):
			if (model == "KROLL" or model == "JPAC"):
				return

		fname = self.GetTxtFileLocTheoryModel(model, ebin, standardE)
		if (model not in self.THEORY_MODELS):
			print
			"ERROR: unexpected model provided!!"
			return

		index_x, index_y, scale_factor_y = -1, -1, 1.
		index_yErr = -1
		if (standardE):
			# if(model=="KROLL" and ebin!=3): return # Nothing provided for other bins yet
			if (model == "ETAMAID"):     index_x, index_y, scale_factor_y, header_catch = 3, 5, 1.e3, "Theta  cosTheta"
			if (model == "JPAC"):        index_x, index_y, scale_factor_y, header_catch = 0, 2, 1.e3, "Dsig/Dt        Dsig/DOmega"
			if (model == "LAGET"):       scale_factor_y = 1e3  # Index x,y not needed for ROOT file
			if (model == "KROLL"):
				index_x, index_y, index_yErr = self.GetKrollTxtIndices(ebin)
				scale_factor_y, header_catch = 1., "-t[GeV^2]"
		if (not standardE):
			if (model == "KROLL"):       return  # Nothing provided for other bins yet
			if (model == "ETAMAID"):     index_x, index_y, scale_factor_y, header_catch = 2, 4, 1.e3, "Theta  cosTheta"
			if (
					model == "JPAC"):        index_x, index_y, scale_factor_y, header_catch = 1, 3, 1.e3, "Dsig/Dt        Dsig/DOmega"
			if (model == "LAGET"):       scale_factor_y = 1e3  # Index x,y not needed for ROOT file

		gr = TGraph()  # Graph to put theory curve into

		if (model == "LAGET"):
			gr_input = TFile.Open(fname).Get("Cross section")
			gr_xaxis = jz_tgraph2numpy(gr_input, "X")[
					   120:]  # Skip first few bins (total 10,000). Visually distracting at edges
			gr_yaxis = jz_tgraph2numpy(gr_input, "Y")[
					   120:]  # Skip first few bins (total 10,000). Visually distracting at edges
			for i in range(len(gr_yaxis) - 1):
				# Don't draw beyond |t|>1 GeV^2 by default, otherwise plot point
				if (gr_xaxis[i] > 1. and self.DrawReggeLTOnly): continue
				if (standardE):     gr.SetPoint(i, gr_xaxis[i], gr_yaxis[i] * scale_factor_y)
				if (not standardE):
					delta_t = gr_xaxis[i + 1] - gr_xaxis[i]
					E_gamma = self.EGAM_LOWE_WBINS[ebin]
					W, ct = calc_W_ct_from_E_t(E_gamma, gr_xaxis[i])
					ds_dt = gr_yaxis[i] * scale_factor_y
					ds_dCosTheta = calc_ct_xsec_from_E_t(E_gamma, gr_xaxis[i], delta_t, ds_dt)
					gr.SetPoint(i, ct, ds_dCosTheta)
		elif (model == "JPAC" or model == "ETAMAID"):  # JPAC or ETAMAID
			theory_txt = open(fname, "r")
			theory_X = array('d', [])
			theory_Y = array('d', [])
			storeLine = False
			for line in theory_txt.readlines():
				if (storeLine):
					line_arr = line.split()
					if ("nan" in line_arr): continue  # Skip over any bins with NaNs
					if (standardE):
						# Don't draw beyond |t|>1 GeV^2 by default, otherwise plot point
						if (abs(float(line_arr[index_x])) > 1. and self.DrawReggeLTOnly and model == "JPAC"): continue
						if (abs(float(
							line_arr[index_x])) > 2. and self.DrawReggeLTOnly and model == "ETAMAID"): continue
						theory_X.append(abs(float(line_arr[index_x])))  # Mandelstam |t|
						theory_Y.append(float(line_arr[index_y]) * scale_factor_y)  # dsigma / dt (nb/GeV^2)
					if (not standardE):
						theory_X.append(float(line_arr[index_x]))  # cos(theta)
						theory_Y.append(float(line_arr[index_y]) * scale_factor_y)  # dsigma / domega (nb/sr)
				if (header_catch in line): storeLine = True
			gr = TGraph(len(theory_X), theory_X, theory_Y)
		elif (model == "KROLL"):
			theory_txt = open(fname, "r")
			theory_X = array('d', [])
			theory_Y = array('d', [])
			theory_Yerr = array('d', [])
			storeLine = False
			for line in theory_txt.readlines():
				if (storeLine):
					line_arr = line.split()
					if (float(line_arr[
								  index_y]) < 1e-5): continue  # Skip points where theory curve is zero, otherwise graph plots points and it looks weird
					theory_X.append(abs(float(line_arr[index_x])))
					theory_Y.append(float(line_arr[index_y]) * scale_factor_y)
					theory_Yerr.append(float(line_arr[index_yErr]) * scale_factor_y)
				if (header_catch in line): storeLine = True
			# Done parsing txt file, now create graph
			y_lo_np = np.array(theory_Y) - np.array(theory_Yerr)
			y_hi_np = np.array(theory_Y) + np.array(theory_Yerr)
			gr_len = len(theory_X)
			gr = TGraph(gr_len, theory_X, theory_Y)
			grmin = TGraph(gr_len, theory_X, y_lo_np)
			grmax = TGraph(gr_len, theory_X, y_hi_np)
			grshade = TGraph(2 * gr_len)
			for i in range(0, gr_len):
				grshade.SetPoint(i, theory_X[i], y_hi_np[i])
				grshade.SetPoint(gr_len + i, theory_X[gr_len - i - 1], y_lo_np[gr_len - i - 1])
			# Cosmetics
			grshade.SetFillStyle(1001)
			grshade.SetFillColorAlpha(kMagenta, 0.1)
			grmin.SetLineWidth(1)
			grmax.SetLineWidth(1)
			gr.SetLineWidth(1)
			grmin.SetLineColorAlpha(kMagenta, 0.1)
			grmax.SetLineColorAlpha(kMagenta, 0.1)
			gr.SetLineColorAlpha(kMagenta, 0.35)
			# gr.SetTitle("Handbag Model at E_{#gamma}=8.25 GeV;Mandelstam |t| (GeV^{2}); d#sigma/dt (nb/GeV^{2})")
			# Save to tuple
			self.KROLL_graphs[ebin] = (grshade, grmin, grmax, gr)

		# A few cosmetics
		if (standardE):     gr.SetLineColorAlpha(self.THEORY_MODEL_COLORS[model], 0.8)
		if (not standardE): gr.SetLineColorAlpha(self.THEORY_MODEL_COLORS[model], 0.6)
		gr.SetLineWidth(1)
		gr.SetMarkerColorAlpha(1, 0.0)
		gr.SetNameTitle(model + "_ebin" + str(ebin), "")

		if (standardE):
			if (model == "JPAC"):    self.JPAC_graphs[ebin] = gr
			if (model == "LAGET"):   self.LAGET_graphs[ebin] = gr
			if (model == "ETAMAID"): self.ETAMAID_graphs[ebin] = gr
		# if(model=="KROLL"):   self.KROLL_graphs[ebin]=gr
		if (not standardE):
			if (model == "JPAC"):    self.JPAC_graphsLE[ebin] = gr
			if (model == "LAGET"):   self.LAGET_graphsLE[ebin] = gr
			if (model == "ETAMAID"): self.ETAMAID_graphsLE[ebin] = gr


	def GetTxtFileLocTheoryModel(self, model, ebin, standardE=True):
		if (model not in self.THEORY_MODELS):
			print("WARNING: unexpected model provided to GetTxtFileLocTheoryModel!!")
			return ""
		txt_file = ""

		if(standardE):
			if(model=="LAGET"):                      txt_file = self.THEORY_DIR+"/"+model+"/ebin"+str(ebin)+".root"
			elif(model=="JPAC" or model=="ETAMAID"): txt_file = self.THEORY_DIR+"/"+model+"/ebin"+str(ebin)+".txt"
			elif(model=="KROLL"):
				# txt_file = self.THEORY_DIR+"/KROLL/ebin3_fromEmail.txt"
				if(ebin<=3):                         txt_file = self.THEORY_DIR+"/KROLL/eta-photoproduction-KPK21-1.out"
				if(3<ebin and ebin<=7):              txt_file = self.THEORY_DIR+"/KROLL/eta-photoproduction-KPK21-2.out"
				if(7<ebin and ebin<=9):              txt_file = self.THEORY_DIR+"/KROLL/eta-photoproduction-KPK21-3.out"
		if(not standardE):
			if(model=="KROLL"):                      txt_file = "ERROR"
			elif(model=="LAGET"):                    txt_file = self.THEORY_DIR+"/"+model+"/LE_ebin"+str(ebin)+".root"
			elif(model=="JPAC" or model=="ETAMAID"): txt_file = self.THEORY_DIR+"/"+model+"/LE_ebin"+str(ebin)+".txt"

		if(not os.path.exists(txt_file) and (model != "KROLL" and not standardE)):
			print
			"WARNING could not find theory txt file: " + txt_file
			sys.exit()

		return txt_file

	def GetKrollTxtIndices(self, ebin):
		yval = -1
		if (ebin == 0):  yval = 1
		elif (ebin == 1):yval = 3
		elif (ebin == 2):yval = 5
		elif (ebin == 3):yval = 7
		elif (ebin == 4):yval = 1
		elif (ebin == 5):yval = 3
		elif (ebin == 6):yval = 5
		elif (ebin == 7):yval = 7
		elif (ebin == 8):yval = 1
		elif (ebin == 9):yval = 3
		return 0, yval, yval + 1

	# Defining here to reduce clutter above
	def AddPlottingVars(self):
		self.RUN_LABELS  = {"SP17":"Spring 2017","SP18":"Spring 2018","FA18":"Fall 2018","FA18LE":"Fall 2018 Low E","SP20":"Spring 2020","COMBINED":"Combined Results"}
		self.MODE_LABELS = {"COMBINED":"","gg":"#gamma#gamma","3pi0":"#pi^{0}#pi^{0}#pi^{0}","3piq":"#pi^{+}#pi^{-}#pi^{0}","3piq_DANIEL":"#pi^{+}#pi^{-}#pi^{0} (Daniel, old)","3piq_DANIELNEW":"#pi^{+}#pi^{-}#pi^{0} (Daniel, new)","omega_gpi0":"#gamma#pi^{0}"}
		self.MARKER_COLOR_BY_RUN    = {"3pi0":kBlue, "3piq":kRed ,"gg":kMagenta+4,"COMBINED":kBlack,"omega_gpi0":kDarkGreen}
		self.MARKER_SHAPE_BY_RUN    = {"SP17":23,"SP18":21,"FA18":33,"FA18LE":20,"SP20":27,"COMBINED":20,} # diamond=33 triangle=23 full_circle=20 full_square=21 open_diamond=27
		self.THEORY_MODEL_COLORS    = {"JPAC":kGreen+2,"LAGET":kBlue,"ETAMAID":12,"KROLL":kMagenta} # Theoretical models with implemented predictions
		self.HISTTITLE_DICT = {
			"gg":"#eta candidates;#gamma#gamma inv. mass (GeV); Counts",
			"3piq":"#eta candidates;#pi^{+}#pi^{-}#pi^{0} inv. mass (GeV); Counts",
			"3pi0":"#eta candidates;#pi^{+}#pi^{-}#pi^{0} inv. mass (GeV); Counts",
			"omega_gpi0":"#omega candidates;#gamma#pi^{0} inv. mass (GeV); Counts",
		}
		self.TITLE_EBINS = [
			"#eta Diff. Cross Sec. 6.5 < E_{#gamma} < 7.0 GeV",
			"#eta Diff. Cross Sec. 7.0 < E_{#gamma} < 7.5 GeV",
			"#eta Diff. Cross Sec. 7.5 < E_{#gamma} < 8.0 GeV",
			"#eta Diff. Cross Sec. 8.0 < E_{#gamma} < 8.5 GeV",
			"#eta Diff. Cross Sec. 8.5 < E_{#gamma} < 9.0 GeV",
			"#eta Diff. Cross Sec. 9.0 < E_{#gamma} < 9.5 GeV",
			"#eta Diff. Cross Sec. 9.5 < E_{#gamma} < 10.0 GeV",
			"#eta Diff. Cross Sec. 10.0 < E_{#gamma} < 10.5 GeV",
			"#eta Diff. Cross Sec. 10.5 < E_{#gamma} < 11.0 GeV",
			"#eta Diff. Cross Sec. 11.0 < E_{#gamma} < 11.5 GeV",
		]
		self.TITLE_EBINS_5x2 = [
			"6.5 < E_{#gamma} < 7.0 GeV",
			"7.0 < E_{#gamma} < 7.5 GeV",
			"7.5 < E_{#gamma} < 8.0 GeV",
			"8.0 < E_{#gamma} < 8.5 GeV",
			"8.5 < E_{#gamma} < 9.0 GeV",
			"9.0 < E_{#gamma} < 9.5 GeV",
			"9.5 < E_{#gamma} < 10.0 GeV",
			"10.0 < E_{#gamma} < 10.5 GeV",
			"10.5 < E_{#gamma} < 11.0 GeV",
			"11.0 < E_{#gamma} < 11.5 GeV",
		]
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


def GetLegendSizeByCase(RegionCase,ET_binning,multiGraphs):
	
	#0.1 is lower limit of plot, 0.9 is upper limit (beyond on either side is labeling+whitespace)
	
	if(not ET_binning): return 0.15,0.56,0.50,0.8 
	if(not multiGraphs):
		# if(RegionCase==0):  return 0.15,0.2,0.4,0.4 # For log-y version
		if(RegionCase==0):  return 0.52,0.6,0.88,0.81
		if(RegionCase==1):  return 0.4,0.7,0.8,0.85
		if(RegionCase==2):  return 0.14,0.55,0.52,0.8
		if(RegionCase==3):  return 0.3,0.55,0.8,0.85
	if(multiGraphs):
		if(RegionCase==0):  return 0.6,0.45,0.88,0.81
		if(RegionCase==1):  return 0.4,0.7,0.8,0.85
		if(RegionCase==2):  return 0.14,0.55,0.52,0.8
		if(RegionCase==3):  return 0.3,0.55,0.8,0.85

def extract_from_graphkey(key):
	# formed from mode+"_"+run+"_"+tag+"_"+yvar+"_"+str(ebin)
	key_split = key.split("_")
	mode,run,tag= key_split[0],key_split[1],key_split[2]
	return mode,run,tag

def GetGraphColor(mode,i):
	baseColor=11
	if(mode=="gg"):   baseColor=kMagenta
	if(mode=="3piq"): baseColor=kRed
	if(mode=="3pi0"): baseColor=kBlue

	if(i==0):  
		if(mode=="gg"):                 return baseColor+3
		if(mode=="3pi0"):               return baseColor-7
		if(mode=="3piq"):               return baseColor-3
	if(i==1):  return baseColor+2
	if(i==2):  return baseColor+1
	if(i==3):  return baseColor-1
	if(i==4):  return baseColor-2
	if(i==5):  return baseColor-3
	if(i==6):  return baseColor-4
	if(i==7):  return baseColor-5
	if(i==8):  return baseColor-6
	if(i==7):  
		if(mode=="gg" or mode=="3piq"): return baseColor-7
		if(mode=="3pi0"):               return baseColor+3
	if(i==10): return baseColor-8
	if(i==11): return baseColor-9
	if(i==12): return baseColor-10

	if(mode=="COMBINED"):
		if (i == 0):  return kBlack
		if (i == 1):  return 12
		if (i == 2):  return 13
		if (i == 3):  return 14
		if (i == 4):  return 15
		if (i == 5):  return 16
		if (i == 6):  return 17

	return 1


def calc_W_from_E(E):
	return ((E + 0.938272) ** 2 - E ** 2) ** 0.5


def calc_W_ct_from_E_t(E, t):
	m_p_sq, m_eta_sq = 0.938272 ** 2, 0.547862 ** 2
	W = calc_W_from_E(E)
	my_t = -1 * abs(t)  # Ensure that value is negative, even if |t| supplied

	E_eta = (W ** 2 + m_eta_sq - m_p_sq) / (2. * W)
	p_eta = sqrt(E_eta * E_eta - m_eta_sq)
	p_gamma = (W ** 2 - m_p_sq) / (2. * W)
	p_diff = p_gamma - p_eta
	t0 = m_eta_sq * m_eta_sq / (4. * W ** 2) - p_diff * p_diff

	sin_theta_over_2 = sqrt((t0 - my_t) / (4 * p_gamma * p_eta))
	ct = cos(2 * asin(sin_theta_over_2))
	return W, ct


def calc_ct_xsec_from_E_t(E, t, Delta_t, xsec):
	W, ct_hi = calc_W_ct_from_E_t(E, t - Delta_t / 2.)
	W, ct_avg = calc_W_ct_from_E_t(E, t)
	W, ct_lo = calc_W_ct_from_E_t(E, t + Delta_t / 2.)
	Delta_ct = ct_hi - ct_lo
	xsec_newUnits = xsec * Delta_t / (2 * 3.14159 * Delta_ct)
	return xsec_newUnits


def calc_t_from_W_ct(W, ct):
	m_p_sq, m_eta_sq = 0.938272 ** 2, 0.547862 ** 2
	E_eta = (W ** 2 + m_eta_sq - m_p_sq) / (2. * W)
	p_eta = sqrt(E_eta * E_eta - m_eta_sq)
	p_gamma = (W ** 2 - m_p_sq) / (2. * W)
	p_diff = p_gamma - p_eta
	t0 = m_eta_sq * m_eta_sq / (4. * W ** 2) - p_diff * p_diff
	t = t0 - 2 * p_gamma * p_eta * (1 - ct)
	return t


LOWE_W_DIVIDER = np.array([(2.52+i*0.04) for i in range(0,23+1)]) # Roughly 2.92-5.8. Although we have data+MC beyond in low E runs, no PS flux.
def GetLowETitle(wbin,latex_ver=False):
	wlo,whi = LOWE_W_DIVIDER[wbin],LOWE_W_DIVIDER[wbin+1]
	if(latex_ver): return "$"+str(round(wlo,2))+" < W < "+str(round(whi,2))+"$"
	return "#eta Diff. Cross Sec. "+str(round(wlo,2))+" < W < "+str(round(whi,2))+" GeV"





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
