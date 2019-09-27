import ROOT
import numpy as np
# Formatting purposes
ROOT.gROOT.SetBatch(1)  # makes it faster- doesn't send lots of canvases over x11 connection


path_to_data = '../../data/'

folder15 = 'breynold/user.breynold.data15_13TeV.00284484.physics_Main.DAOD_NTUP_JTRIG_JETM1.r9264_p3083_p3601_j042_tree.root/'
file15 = 'user.breynold.18753218._000001.tree.root'
folder16 = 'breynold/user.breynold.data16_13TeV.00307656.physics_Main.DAOD_NTUP_JTRIG_JETM1.r9264_p3083_p3601_j042_tree.root/'
file16 = 'user.breynold.18797259._000001.tree.root'

# Load a ROOT file
path1 = path_to_data + folder16 + file16
rootfile1 = ROOT.TFile(path1)
ttree1 = rootfile1.Get('outTree/nominal')

c1 = ROOT.TCanvas("c1", "ROOT Canvas")
h1 = ROOT.TH1D("h1", "Invariant mass", 100, 0, 3000)

n_events = ttree1.GetEntries()
for ii in np.arange(ttree1.GetEntriesFast()):
    ttree1.GetEntry(ii)
    if ii % 100000 == 0:
        print('Entry ', ii, ' done...')

    if len(ttree1.AntiKt4EMPFlowJets_Calib2018_E) > 1:  # Only consider events with 2 or more jets
        e1 = ttree1.AntiKt4EMPFlowJets_Calib2018_E[0]
        pt1 = ttree1.AntiKt4EMPFlowJets_Calib2018_pt[0]
        eta1 = ttree1.AntiKt4EMPFlowJets_Calib2018_eta[0]
        phi1 = ttree1.AntiKt4EMPFlowJets_Calib2018_phi[0]

        e2 = ttree1.AntiKt4EMPFlowJets_Calib2018_E[1]
        pt2 = ttree1.AntiKt4EMPFlowJets_Calib2018_pt[1]
        eta2 = ttree1.AntiKt4EMPFlowJets_Calib2018_eta[1]
        phi2 = ttree1.AntiKt4EMPFlowJets_Calib2018_phi[1]


        v1 = ROOT.TLorentzVector()
        v1.SetPtEtaPhiE(pt1, eta1, phi1, e1)
        v2 = ROOT.TLorentzVector()
        v2.SetPtEtaPhiE(pt2, eta2, phi2, e2)
        vjj = v1 + v2

        mjj = vjj.M()

        h1.Fill(mjj)

h1.GetXaxis().SetTitle('mjj')
h1.GetYaxis().SetTitle('Number of events')
c1.SetGrid()
h1.Draw()
c1.Draw()
c1.SaveAs("./dijet_mass_plot.png")
