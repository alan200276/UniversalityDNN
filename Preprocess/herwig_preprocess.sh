#/bin/bash

HOMEPATH="/home/alan/MC_Samples/Universality_DNN"
datapath="/home/u5/Universality/Herwig_angular"
outpath="Out"
path="Preprocess"

date

echo "Start Running"


date +"%Y %b %m"



echo "W Herwig Angular"
python3 $HOMEPATH/$path/preprocess.py $datapath/ppwz_herwig_ang_1.root 0 > $HOMEPATH/$outpath/Herwig_angular_W/preprocess_ppwz_herwig_ang_1.log


echo "QCD Herwig Angular"
python3 $HOMEPATH/$path/preprocess.py $datapath/ppjj_herwig_ang_1.root 1 > $HOMEPATH/$outpath/Herwig_angular_QCD/preprocess_ppjj_herwig_ang_1.log




date +"%Y %b %m"
   
date +"%Y %b %m"
date +"%r"


echo "Finish"

date