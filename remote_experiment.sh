#!/bin/bash

# runs the experiment (Nevergrad) on a remote machine


# 1) run the cluster/manage.py upload MACHINE_IP
# 2) login to the machine
# 3) prepare the machine and run this script:

# rm -rf distributed_es && mkdir distributed_es && tar -xvf /home/ubuntu/source.tar -C /home/ubuntu/ && cd distributed_es && export PYTHONPATH=. && ./remote_experiment.sh


tmux new-session -d -s "exp" 'export PYTHONPATH=. && python nevergrad_es/experiment.py with ant tool=CMA num_episodes=5'

