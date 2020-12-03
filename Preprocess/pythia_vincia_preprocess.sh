#/bin/bash

HOMEPATH="/home/alan/MC_Samples/Universality_DNN"
datapath="/home/u5/Universality/Pythia_vincia"
outpath="Out"
path="Preprocess"

date

echo "Start Running"


date +"%Y %b %m"



echo "W Pythiav Vincia"
python3 $HOMEPATH/$path/preprocess.py $datapath/ppwz_pythia_vin_1.root 0 > $HOMEPATH/$outpath/Pythia_vincia_W/preprocess_ppwz_pythia_vin_1.log


echo "QCD Pythia Vincia"
python3 $HOMEPATH/$path/preprocess.py $datapath/ppjj_pythia_vin_1.root 1 > $HOMEPATH/$outpath/Pythia_vincia_QCD/preprocess_ppjj_pythia_vin_1.log




date +"%Y %b %m"
   
date +"%Y %b %m"
date +"%r"


echo "Finish"

date