#!/bin/bash

validation_interval=5

max_epochs=()

for ii in {1..20}
do
   max_epochs[${#max_epochs[@]}]=$((validation_interval*ii))
done

max_epoch=10



for (( c=$((max_epoch-validation_interval+1)); c<=$max_epoch; c++ ))
do 
   echo "Welcome $c times"
done

if [ -f "../fairseq_mt_data.ipynb" ]; then
    echo " exists."
fi