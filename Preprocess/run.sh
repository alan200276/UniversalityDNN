#/bin/bash


date

echo "Start Submission"


# nohup bash -e sherpa_preprocess_separate.sh > jetimage_sherpa_def.out &
# sleep 10
# nohup bash -e pythia_default_preprocess_separate.sh > nohup_pythia_def.log &
# sleep 10
# nohup bash -e pythia_dipole_preprocess_separate.sh > jetimage_pythia_dip.out &
# sleep 10
# nohup bash -e pythia_vincia_preprocess_separate.sh > jetimage_pythia_vin.out &
# sleep 10
# nohup bash -e herwig_preprocess_separate.sh > jetimage_herwig_ang.out &

bash -e sherpa_preprocess_separate.sh #> jetimage_sherpa_def.out 
# sleep 10
bash -e pythia_default_preprocess_separate.sh #> nohup_pythia_def.log 
# sleep 10
bash -e pythia_dipole_preprocess_separate.sh #> jetimage_pythia_dip.out 
# sleep 10
bash -e pythia_vincia_preprocess_separate.sh #> jetimage_pythia_vin.out 
# sleep 10
bash -e herwig_preprocess_separate.sh #> jetimage_herwig_ang.out 





echo "Job Submitted"

date