#!/bin/bash
cd $PWD
export data_file_id=$(cut -d "=" -f2 <<< $1)
export weights_file_id=$(cut -d "=" -f2 <<< $2)
export data_file_name=data_road.zip
export weights_file_name="weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

download_data () {
    echo "downloading data"
    wget -q --show-progress --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$data_file_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
    wget -q --show-progress --load-cookies cookies.txt -O $data_file_name 'https://docs.google.com/uc?export=download&id='$data_file_id'&confirm='$(<confirm.txt)
    unzip $data_file_name
    mv $data_file_name data
    rm $data_file_name cookies.txt confirm.txt
}

download_weights () {
    mkdir -p weights
    echo "downloading weights file"
    wget -q --show-progress --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$weights_file_id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
    wget -q --show-progress --load-cookies cookies.txt -O $weights_file_name 'https://docs.google.com/uc?export=download&id='$weights_file_id'&confirm='$(<confirm.txt)
    rm cookies.txt confirm.txt
}

echo "loading from google drive"
if [ ! -d "data" ] # if data folder doesnt exist
then # load data from gdrive
    download_data
elif [ ! "$(ls -A data)" ] # data folder exists and is empty
then # load data from gdrive
    download_data
else
    echo "data exists"
fi

if [ ! -d "weights" ] # if weights folder doesnt exist
then # load data from gdrive
    download_weights
elif [ ! "$(ls -A weights)" ] # data folder exists and is empty
then # load data from gdrive
    download_weights
else
    echo "weights file exists"
fi








