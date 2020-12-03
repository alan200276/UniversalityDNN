#/bin/bash

HOMEPATH="/home/alan/MC_Samples/Universality_DNN"
datapath="/home/u5/Universality/"
outpath="Out"
path="Preprocess"

date

echo "Start Running"


date +"%Y %b %m"



echo "W Herwig Angular"
python3 $HOMEPATH/$path/preprocess.py $datapath/Sherpa_ppwz/ppwz_sherpa_def_1.root 0 > $HOMEPATH/$outpath/Sherpa_W/preprocess_ppwz_sherpa_def_1_1.log


echo "QCD Herwig Angular"
python3 $HOMEPATH/$path/preprocess.py $datapath/Sherpa_ppjj/ppjj_sherpa_def_1.root 1 > $HOMEPATH/$outpath/Sherpa_QCD/preprocess_ppjj_sherpa_def_1_1.log




date +"%Y %b %m"
   
date +"%Y %b %m"
date +"%r"


echo "Finish"

date