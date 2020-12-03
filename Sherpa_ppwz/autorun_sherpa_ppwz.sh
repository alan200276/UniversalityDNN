#!/bin/bash

outpath="/home/alan/MC_Samples/Universality_DNN/Out/Sherpa_W"
mcdatapath="/home/u5"

echo "Start Running"
date


nevent=100000

i=2
while [ $i != 11 ]
do
    
    rand=$RANDOM

    echo "Random Seed =  $rand"
    sed -i -e "s/def_"$(($i-1))"/def_"$i"/g" ./PPWZ.dat 

    /usr/local/bin/Sherpa -f ./PPWZ.dat -R $rand EVENTS=$nevent > $outpath/ppwz_sherpa_def_"$i".log 

   date +"%Y %b %m"
   date +"%r"
   i=$(($i+1))

done

sed -i -e "s/def_"$(($i-1))"/def_0/g" ./PPWZ.dat 


echo "Start UnZip < .hepmc2g >"
date

i=2
while [ $i != 11 ]
do
   echo i=$i
   gunzip < $mcdatapath/Universality/Sherpa_ppwz/ppwz_sherpa_def_"$i".hepmc2g > $mcdatapath/Universality/Sherpa_ppwz/ppwz_sherpa_def_"$i".hepmc 
     
   date +"%Y %b %m"
   date +"%r"
   i=$(($i+1))

done

echo "Finish"

date