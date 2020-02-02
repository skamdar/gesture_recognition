#!/bin/bash
##name of the job
#PBS -N gr
##job output log
#PBS -o out.log
##job/application error logs
#PBS -e error.log
##requesting number of nodes and resources
#PBS -l nodes=1:ppn=20
##selecting queue
#PBS -q prerungs
cd $PBS_O_WORKDIR

#module load utils/anaconda3.5

##python main.py --root_path /home/gcnandi/Sonu/data --video_path grit/jpg --annotation_path grit.json --result_path results --dataset grit --n_classes 400 --n_finetune_classes 9 --pretrain_path models/resnet-18-kinetics.pth --ft_begin_index 4 --model resnet --model_depth 18 --resnet_shortcut A --batch_size 32 --n_threads 4 --checkpoint 10

python main2.py --root_path /home/gcnandi/Sonu/data --video_path grit/jpg --annotation_path grit.json \
--result_path results --dataset grit --n_classes 9 --model mymodel --batch_size 5 --n_threads 4 --checkpoint 10 --n_epochs 100
