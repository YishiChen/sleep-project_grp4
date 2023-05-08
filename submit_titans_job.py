import os
import tempfile

# fmt: off
JOBS = [
    {
        "jobname": "detr",
        "partition": "titans",
        "reservation": "comp-gpu05",  # This is my GPU node, comment this line and remove line 27, if you wish to send the job out to all nodes
        "time": "4-00:00:00",  # Days-Hours:Minutes:Seconds
        "ncpus": 2,  # Number of CPU cores
        "gpus": 1,  # Number of GPUs
        "memory": "256G",  # This is total RAM, change this accordingly to use
        "command": "python -m torch.distributed.launch --nproc_per_node=4  --use_env detr-main/main.py --resume /scratch/s203877/checkpoints/3365/checkpoint0019.pth",
        "log_path": "/scratch/s203877/runs"  # Usually this is your scratch space

    },
]
# fmt: on


def submit_job(jobname, partition, time, reservation, ncpus, gpus, command, memory, log_path, *args):
    content = f"""#!/bin/bash
#
#SBATCH --job-name={jobname}
#SBATCH --time={time}
#SBATCH -p {partition}
#SBATCH --cpus-per-task={ncpus}
#SBATCH --gres=gpu:{gpus}
#SBATCH --ntasks-per-node={gpus}
#SBATCH -w {reservation}
#SBATCH --mem={memory}
#SBATCH --output={log_path}/{jobname}.out
#SBATCH --error={log_path}/{jobname}.err
##################################################

# Change this to correct directory
cd $HOME/bachelor/sleep-project_grp4

# Activate conda
# source $GROUP_HOME/miniconda3/bin/activate
source $GROUP_HOME/opt/miniconda3/bin/activate

# Activate correct conda environment
conda activate mros

# Run command
{command}
"""
    print(content)
    with tempfile.NamedTemporaryFile(delete=False) as j:
        j.write(content.encode())
    os.system(f"sbatch {j.name}")


if __name__ == "__main__":

    print(f"Submitting {len(JOBS)} job(s) ...")
    for jobinfo in JOBS:
        submit_job(**jobinfo)

    print("All jobs have been submitted!")
