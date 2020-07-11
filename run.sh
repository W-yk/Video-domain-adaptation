#!/bin/bash

#====== parameters ======#

dataset=hmdb_ucf # hmdb_ucf | hmdb_ucf_small | ucf_olympic
class_file='data/classInd_'$dataset'.txt'

#====== select dataset ======#
path_data_root=/slwork/wyk/TA3N/dataset/ # depend on users
path_exp_root=action-experiments/ # depend on users

if [ "$dataset" == "hmdb_ucf" ] || [ "$dataset" == "hmdb_ucf_small" ] ||[ "$dataset" == "ucf_olympic" ]
then
	dataset_source=ucf101 # depend on usersf
	dataset_target=hmdb51 # depend on users
	dataset_val=hmdb51 # depend on users
	num_source=1438 # number of training data (source) 
	num_target=840 # number of training data (target)

	path_data_source=$path_data_root$dataset_source'/'
	path_data_target=$path_data_root$dataset_target'/'
	path_data_val=$path_data_root$dataset_val'/'

	if [[ "$dataset_source" =~ "train" ]]
	then
		dataset_source=$dataset_source
	else
    	dataset_source=$dataset_source'_train'
	fi

	if [[ "$dataset_target" =~ "train" ]]
	then
		dataset_target=$dataset_target
	else
    	dataset_target=$dataset_target'_train'
	fi
	
	if [[ "$dataset_val" =~ "val" ]]
	then
		dataset_val=$dataset_val
	else
    	dataset_val=$dataset_val'_val'
	fi

	train_source_list=$path_data_source'list_'$dataset_source'_'$dataset'-'feature'.txt'
	train_target_list=$path_data_target'list_'$dataset_target'_'$dataset'-'feature'.txt'
	val_list=$path_data_val'list_'$dataset_val'_'$dataset'-'feature'.txt'

fi

python ResNetCRNN.py  --train_source_list $train_source_list --train_target_list $train_target_list --val_list $val_list --class_file $class_file
