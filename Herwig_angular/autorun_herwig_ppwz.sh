#!/bin/bash

outpath="/home/alan/MC_Samples/Universality_DNN/Out/Herwig_angular_W"
mcdatapath="/home/u5"

echo "Start Running"
date


nevent=100000

i=10
while [ $i != 11 ]
do
    rand=$RANDOM
    
    echo "Random Seed =  $rand "    
    
    /herwig/bin/Herwig read ./ppwz_angular.in 
    
    /herwig/bin/Herwig run ./ppwz_angular.run -N $nevent -s $rand -d 1 > $outpath/ppwz_angular_"$i".log 
    
    
   date +"%Y %b %m"
   date +"%r"
   i=$(($i+1))
   
   
   if [ "$i" < "10" ];then

       sed -i -e "s/run_0"$(($i-1))"/run_0"$(($i))"/g" ./ppwz_angular.in  

       sed -i -e "s/ppwz_angular_"$(($i-1))"/ppwz_angular_"$(($i))"/g" ./ppwz_angular.in
       
    elif [ "$i" == "10" ];then

       sed -i -e "s/run_0"$(($i-1))"/run_"$(($i))"/g" ./ppwz_angular.in  

       sed -i -e "s/ppwz_angular_"$(($i-1))"/ppwz_angular_"$(($i))"/g" ./ppwz_angular.in

    elif [ "$i" > "10" ];then

       sed -i -e "s/run_"$(($i-1))"/run_"$(($i))"/g" ./ppwz_angular.in  

       sed -i -e "s/ppwz_angular_"$(($i-1))"/ppwz_angular_"$(($i))"/g" ./ppwz_angular.in
        
    fi

  

done

sed -i -e "s/run_"$(($i))"/run_01/g" ./ppwz_angular.in  
sed -i -e "s/ppwz_angular_"$(($i))"/ppwz_angular_1/g" ./ppwz_angular.in


echo "Finish"

date