#!/bin/bash

echo "Start Running"

outpath="/home/alan/MC_Samples/Universality_DNN/Out/Sherpa_W"
hepmcpath="/home/u5/Universality/Sherpa_ppwz"
savepath="/home/u5/Universality/Sherpa_ppwz"

filename="ppwz_sherpa_def"

i=2

while [ $i != 11 ]
do

       date +"%Y %b %m"
       date +"%r"
       echo "PPWZ"
       
       cd /root/MG5_aMC_v2_7_2/Delphes
       
        nohup ./DelphesHepMC ./cards/delphes_card_ATLAS.tcl $savepath/"$filename"_$i.root $hepmcpath/"$filename"_$i.hepmc > $outpath/"$filename"_"$i"_log.txt &

       date +"%Y %b %m"
       date +"%r"

   
#    date +"%Y %b %m"
#    date +"%r"
   i=$(($i+1))

done

echo "Finish"

date
