#!/bin/bash

# runs the experiment (Nevergrad) on a remote machine

# 1) run the cluster/manage.py upload MACHINE_IP
# 2) login to the machine
# 3) run the experiment: ./remote_experiment.sh

rm -rf distributed_es && mkdir distributed_es && tar -xvf /home/ubuntu/source.tar -C /home/ubuntu/ && cd distributed_es

tmux new-session -d -s "exp" 'export PYTHONPATH=. && python nevergrad_es/experiment.py with pendulum_partial tool=CMA num_episodes=30'

