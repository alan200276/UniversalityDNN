#!/bin/bash

echo "Start Running"

outpath="/home/alan/MC_Samples/Universality_DNN/Out/Sherpa_QCD"
hepmcpath="/home/u5/Universality/Sherpa_ppjj"
savepath="/home/u5/Universality/Sherpa_ppjj"

filename="ppjj_sherpa_def"

i=11

while [ $i != 21 ]
do

       date +"%Y %b %m"
       date +"%r"
       echo "PPJJ"
       echo "$i"
       
#        cd /root/MG5_aMC_v2_7_2/Delphes
       cd /root/Delphes-3.4.2
       
        nohup ./DelphesHepMC ./cards/delphes_card_ATLAS.tcl $savepath/"$filename"_$i.root $hepmcpath/"$filename"_$i.hepmc > $outpath/"$filename"_"$i"_log.txt &

       date +"%Y %b %m"
       date +"%r"

   
#    date +"%Y %b %m"
#    date +"%r"
   i=$(($i+1))

done

echo "Finish"

date
