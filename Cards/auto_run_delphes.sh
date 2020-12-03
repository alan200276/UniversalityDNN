#!/bin/bash

echo "Start Running"

cmndpath="/home/alan/MC_Samples/Universality_DNN/Pythia/ppzz.cmnd"
cmscardpath="/home/alan/MC_Samples/Universality_DNN/Pythia/delphes_card_CMS.dat"
savepath="/home/alan/MC_Samples/Universality_DNN/Pythia"
filename="ppzz_Simple"

i=0
while [ $i != 1 ]
do

       date +"%Y %b %m"
       date +"%r"
       echo "PPWZ"

#        nohup  &
       ./DelphesPythia8 $cmscardpath $cmndpath $savepath/"$filename".root > $savepath/"$filename"_log.txt

       date +"%Y %b %m"
       date +"%r"

   
   date +"%Y %b %m"
   date +"%r"
   i=$(($i+1))

done

echo "Finish"

date
