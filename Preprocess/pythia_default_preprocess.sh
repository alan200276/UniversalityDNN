#/bin/bash

PROCESS="Pythia_default"
HOMEPATH="/home/alan/MC_Samples/Universality_DNN"
datapath="/home/u5/Universality/"$PROCESS
outpath_wz="Out/"$PROCESS_W
outpath_jj="Out/"$PROCESS"_QCD"
process="pythia_def"

date

echo "Start Running"


date +"%Y %b %m"


i=1
while [ $i != 3 ]
do 

    date +"%r"
    
    if [ "$i" == "1" ];then
        
        echo "W $PROCESS"
        python3 $HOMEPATH/Preprocess/preprocess.py $datapath/ppwz_"$process"_"$i".root 0 > $HOMEPATH/$outpath_wz/preprocess_ppwz_"$process"_"$i".log
        
        echo "QCD $PROCESS"
        python3 $HOMEPATH/Preprocess/preprocess.py $datapath/ppjj_"$process"_"$i".root 0 > $HOMEPATH/$outpath_jj/preprocess_ppjj_"$process"_"$i".log
    
    elif [ "$i" != "1" ];then

        echo "W $PROCESS"
        python3 $HOMEPATH/Preprocess/preprocess.py $datapath/ppwz_"$process"_"$i".root 1 > $HOMEPATH/$outpath_wz/preprocess_ppwz_"$process"_"$i".log
        
        echo "QCD $PROCESS"
        python3 $HOMEPATH/Preprocess/preprocess.py $datapath/ppjj_"$process"_"$i".root 1 > $HOMEPATH/$outpath_jj/preprocess_ppjj_"$process"_"$i".log

    fi


    date +"%Y %b %m"
    date +"%r"
    i=$(($i+1))

done



echo "Finish"

date