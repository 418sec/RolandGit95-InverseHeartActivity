#! python

# name
#$ -N multicondition

# execute from current directory
#$ -cwd

# Preserve environment variables
#$ -V

# Provide path to python executable
#$ -S /home/stenger/smaxxhome/anaconda3/envs/dolfin/bin/python

# Merge error and out
#$ -j yes

# Path for output
#$ -o /home/stenger/smaxxhome/Masterthesis/heartmulticondition-master/2020-01-28_tests/outputs

# Limit memory to <64G/16
#$ -hard -l h_vmem=3.5G

# serial queue
#$ -q grannus.q

# job array of length 1
#$ -t 1-2

import os
from main import trainOnDataset, defineModel

if __name__=='__main__':
    print(int(os.getenv('SGE_TASK_ID', -1)))
    identifier = os.getenv('SGE_TASK_ID', 'TASK_ID') + '_' + os.getenv('JOB_ID', 'JOB_ID')
    
    model = defineModel(feature_dim=(200, 70), output_dim=3)
    trainOnDataset(model, savename='test1')