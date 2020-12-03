#!/bin/bash

echo "Start Running"

outpath="/home/alan/MC_Samples/Universality_DNN/Out/Pythia_dipole_QCD"
hepmcpath="/home/u5/Universality/Pythia_dipole"
savepath="/home/u5/Universality/Pythia_dipole"
rootfilename="ppjj_pythia_dip"
hepmcfilename="ppjj_pythia_dip"


i=11

while [ $i != 21 ]
do

       date +"%Y %b %m"
       date +"%r"
       echo "PPJJ"
       echo "$i"
       
#        cd /root/MG5_aMC_v2_7_2/Delphes
       cd /root/Delphes-3.4.2
       
        nohup ./DelphesHepMC ./cards/delphes_card_ATLAS.tcl $savepath/"$rootfilename"_$i.root $hepmcpath/"$hepmcfilename"_$i.hepmc > $outpath/"$rootfilename"_"$i"_log.txt &
        
       date +"%Y %b %m"
       date +"%r"

   
   date +"%Y %b %m"
   date +"%r"
   i=$(($i+1))

done

echo "Finish"

date
