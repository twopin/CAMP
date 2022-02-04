# this shell script provides an easy way to apply CAMP on deciphering peptide-protein interactions
# make sure that pip can work on your server

unzip example_data_feature.zip
unzip ./model/CAMP.h5.zip
unzip ./model/CAMP_BS.h5.zip

mkdir example_prediction

python -u predict_CAMP.py


