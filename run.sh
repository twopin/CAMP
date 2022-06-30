# this shell script provides an easy way to apply CAMP on deciphering peptide-protein interactions
# make sure that pip can work on your server


#pip install Keras==2.0.8
#pip install tensorflow=1.2.1

unzip example_data_feature.zip
unzip ./model/CAMP.h5.zip
unzip ./model/CAMP_BS.h5.zip

mkdir -p example_prediction

python -u predict_CAMP.py


