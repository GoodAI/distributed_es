#!/bin/bash

# runs experiment on 5 clusters in parallel

python cluster/launch.py launch --size=14 --workers_per_machine=8 --config='pendulum_partial pop_size=200'
python cluster/launch.py launch --size=14 --workers_per_machine=8 --config='pendulum_partial pop_size=200'
python cluster/launch.py launch --size=14 --workers_per_machine=8 --config='pendulum_partial pop_size=200'
python cluster/launch.py launch --size=14 --workers_per_machine=8 --config='pendulum_partial pop_size=200'
python cluster/launch.py launch --size=14 --workers_per_machine=8 --config='pendulum_partial pop_size=200'

