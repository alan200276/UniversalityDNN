#/bin/bash

PROCESS="Pythia_vincia"
HOMEPATH="/home/alan/MC_Samples/Universality_DNN"
datapath="/home/u5/Universality/"$PROCESS
outpath_wz="Out/"$PROCESS"_W"
outpath_jj="Out/"$PROCESS"_QCD"
process="pythia_vin"

date

echo "Start Running"


date +"%Y %b %m"


i=4
while [ $i != 6 ]
do 

    echo "W $PROCESS"
    nohup python3 $HOMEPATH/Preprocess/preprocess_separate.py $datapath/ppwz_"$process"_"$i".root 0 $i > $HOMEPATH/$outpath_wz/preprocess_ppwz_"$process"_"$i".log &
        
    echo "QCD $PROCESS"
    nohup python3 $HOMEPATH/Preprocess/preprocess_separate.py $datapath/ppjj_"$process"_"$i".root 0 $i > $HOMEPATH/$outpath_jj/preprocess_ppjj_"$process"_"$i".log &
    
    
#     echo "W Jet Images $PROCESS"
#     python3 $HOMEPATH/Preprocess/Jet_Images.py $datapath/ppwz_"$process"_"$i".root 0 $i > $HOMEPATH/$outpath_wz/Jet_Images_ppwz_"$process"_"$i".log 
    
#     echo "QCD Jet Images $PROCESS"
#     python3 $HOMEPATH/Preprocess/Jet_Images.py $datapath/ppjj_"$process"_"$i".root 0 $i > $HOMEPATH/$outpath_jj/Jet_Images_ppjj_"$process"_"$i".log 

    date +"%Y %b %m"
    date +"%r"
    i=$(($i+1))

done



echo "Finish"

date
