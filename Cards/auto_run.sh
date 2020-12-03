#!/bin/bash

cardpath="/home/alan/MC_Samples/Universality_DNN/Cards"

outpath="/home/alan/MC_Samples/Universality_DNN/Out"

mcdatapath="/home/u5"


echo "Start Running"

i=20
while [ $i != 21 ]
do
   echo i=$i

   date +"%Y %b %m"
   date +"%r"
   
#    echo "PPWZ"
#    python /root/MG5_aMC_v2_7_2/bin/mg5_aMC $cardpath/ppwz.txt > $outpath/ppwz_"$i".log 
   
   
   echo "PPjj"
#    python /root/MG5_aMC_v2_7_2/bin/mg5_aMC $cardpath/ppjj.txt > $outpath/ppjj_"$i".log
   python /root/MG5_aMC_v2_7_3/bin/mg5_aMC $cardpath/ppjj.txt > $outpath/ppjj_"$i".log
   
   
   

   date +"%Y %b %m"
   date +"%r"
   i=$(($i+1))

done

gzip -d $mcdatapath/pp*/Events/run_*/unweighted_events.lhe.gz


echo "Finish"

date
