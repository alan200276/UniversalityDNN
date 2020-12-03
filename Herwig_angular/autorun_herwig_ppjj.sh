#!/bin/bash

outpath="/home/alan/MC_Samples/Universality_DNN/Out/Herwig_angular_QCD"
mcdatapath="/home/u5"

echo "Start Running"
date

lhapdf install NNPDF23_nlo_as_0118

nevent=100000

i=12
while [ $i != 21 ]
do
    rand=$RANDOM
    
    echo "Random Seed =  $rand "   
    
    echo "Reading"   
    
    nohup /herwig/bin/Herwig read ./ppjj_angular.in > read_"$i"_out &
    
    sleep 10
    
    echo "Running"   
    
    nohup /herwig/bin/Herwig run ./ppjj_angular.run -N $nevent -s $rand -d 1 > $outpath/ppjj_angular_"$i".log &
    
    sleep 5
    
   date +"%Y %b %m"
   date +"%r"
   i=$(($i+1))
   
   
   if [ "$i" -lt "10" ];then

       sed -i -e "s/run_0"$(($i-1))"/run_0"$(($i))"/g" ./ppjj_angular.in  

       sed -i -e "s/ppjj_angular_"$(($i-1))"/ppjj_angular_"$(($i))"/g" ./ppjj_angular.in
       
    elif [ "$i" -eq "10" ];then

       sed -i -e "s/run_0"$(($i-1))"/run_"$(($i))"/g" ./ppjj_angular.in  

       sed -i -e "s/ppjj_angular_"$(($i-1))"/ppjj_angular_"$(($i))"/g" ./ppjj_angular.in

    elif [ "$i" -gt "10" ];then

       sed -i -e "s/run_"$(($i-1))"/run_"$(($i))"/g" ./ppjj_angular.in  

       sed -i -e "s/ppjj_angular_"$(($i-1))"/ppjj_angular_"$(($i))"/g" ./ppjj_angular.in
        
    fi

  

done


if [ "$i" -lt "10" ];then

    sed -i -e "s/run_0"$(($i))"/run_01/g" ./ppjj_angular.in 
    
elif [ "$i" -eq "10" ];then

    sed -i -e "s/run_"$(($i))"/run_01/g" ./ppjj_angular.in
    
elif [ "$i" -gt "10" ];then

    sed -i -e "s/run_"$(($i))"/run_01/g" ./ppjj_angular.in
    
fi

sed -i -e "s/ppjj_angular_"$(($i))"/ppjj_angular_1/g" ./ppjj_angular.in


echo "Finish"

date