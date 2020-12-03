#!/bin/bash

echo "Start Running"

outpath="/home/alan/MC_Samples/Universality_DNN/Out/Pythia_vincia_W"
hepmcpath="/home/u5/Universality/Pythia_vincia"
savepath="/home/u5/Universality/Pythia_vincia"
rootfilename="ppwz_pythia_vin"
hepmcfilename="ppwz_pythia_vin"


i=2

while [ $i != 11 ]
do

       date +"%Y %b %m"
       date +"%r"
       echo "PPWZ"
       
       cd /root/MG5_aMC_v2_7_2/Delphes
       
        nohup ./DelphesHepMC ./cards/delphes_card_ATLAS.tcl $savepath/"$rootfilename"_$i.root $hepmcpath/"$hepmcfilename"_$i.hepmc > $outpath/"$rootfilename"_"$i"_log.txt &
        
       date +"%Y %b %m"
       date +"%r"

   
   date +"%Y %b %m"
   date +"%r"
   i=$(($i+1))

done

echo "Finish"

date
