#/bin/bash

HOMEPATH="/home/alan/MC_Samples/Universality_DNN"
datapath="/home/u5/Universality/Pythia_dipole"
outpath="Out"
path="Preprocess"

date

echo "Start Running"


date +"%Y %b %m"



echo "W Pythia Dipole"
python3 $HOMEPATH/$path/preprocess.py $datapath/ppwz_pythia_dip_1.root 0 > $HOMEPATH/$outpath/Pythia_dipole_W/preprocess_ppwz_pythia_dip_1.log


echo "QCD Pythia Dipole"
python3 $HOMEPATH/$path/preprocess.py $datapath/ppjj_pythia_dip_1.root 1 > $HOMEPATH/$outpath/Pythia_dipole_QCD/preprocess_ppjj_pythia_dip_1.log




date +"%Y %b %m"
   
date +"%Y %b %m"
date +"%r"


echo "Finish"

date