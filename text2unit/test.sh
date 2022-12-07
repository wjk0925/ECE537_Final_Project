#!/bin/bash
for (( epoch=0; epoch<=295; epoch+=5 ))
do
        echo "evaluating epoch${epoch}"
        #./eval_t2u.sh --project_dir ${project_dir} --epoch ${epoch} --gen_subset valid --dataset ${dataset} --t2u_dir ${t2u_dir}
done
