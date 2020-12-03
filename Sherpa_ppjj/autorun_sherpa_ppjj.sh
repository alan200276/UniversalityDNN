#!/bin/bash

outpath="/home/alan/MC_Samples/Universality_DNN/Out/Sherpa_QCD"
mcdatapath="/home/u5"

echo "Start Running"
date


# nevent=100000

# i=11
# while [ $i != 21 ]
# do

#     rand=$RANDOM
    
#     echo i=$i
#     echo "Random Seed =  $rand"

#     nohup /usr/local/bin/Sherpa -f ./PPJJ.dat -R $rand EVENTS=$nevent > $outpath/ppjj_sherpa_def_"$i".log &
#     sleep 5
    
    
#     date +"%Y %b %m"
#     date +"%r"
#     i=$(($i+1))

#     sed -i -e "s/def_"$(($i-1))"/def_"$i"/g" ./PPJJ.dat 


# done

# sed -i -e "s/def_"$(($i))"/def_0/g" ./PPJJ.dat 


echo "Start UnZip < .hepmc2g >"
date

i=11
while [ $i != 21 ]
do
   echo i=$i
   gunzip < $mcdatapath/Universality/Sherpa_ppjj/ppjj_sherpa_def_"$i".hepmc2g > $mcdatapath/Universality/Sherpa_ppjj/ppjj_sherpa_def_"$i".hepmc 
     
   date +"%Y %b %m"
   date +"%r"
   i=$(($i+1))

done

echo "Finish"

date