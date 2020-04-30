import base64
import pathlib
import random
import tarfile
from datetime import datetime
from typing import Tuple, List, Optional

import boto3
import click

from default_params import AUTHOR
from es.es_utils.es_common import REMOTE_MASTER_SOCKET_PATH

""" ------------- fill in these values ------------ """
AWS_KEY_NAME = 'your_aws_key_name'
DEFAULT_IMAGE_ID = 'ami-XXXXX'
PEM_FILE = '~/your.pem'
DEFAULT_USER_ID = 'AIDAXXXXXXXXXXXXXXX'
DEFAULT_ROLE_ID = 'AROAQXXXXXXXXX'
DEFAULT_SECURITY_GROUP = 'sg-XXXXX'

OWNER = AUTHOR
GROUP_NAME = 'es_group'

REDIS_CLUSTER_CODE_ACCESS = 'redis_cluster_code_access'
DEFAULT_BUCKET_NAME = 'redis-cluster-storage'
AVAILABILITY_ZONES = ["eu-west-1a", "eu-west-1b"]

ES_ROLE_KEY = 'ES_role'
TAR_NAME = 'source.tar'
CLUSTER_NAME_KEY = 'cluster_name'


@click.group()
def cli():
    pass


def up(bucket_name: str, key_name: str, filename: str):
    s3_res = boto3.resource('s3')
    s3_res.Bucket(bucket_name).upload_file(Filename=filename, Key=key_name)
    remote_path = f'{bucket_name}/{key_name}'
    print(f'done, uploaded to: {remote_path}, obtaining the url')


def down(bucket_name: str, filename_key: str, destination: str):
    s3_res = boto3.resource('s3')
    print(f'Downloading ---{filename_key}--- to: {destination}')
    s3_res.Bucket(bucket_name).download_file(Key=filename_key, Filename=destination)
    print(f'done!')


def get_tags(owner: str, cluster_name: str, storage_name: str) -> Tuple[List, List]:
    common_tags = [
        {"Key": 'Owner', "Value": owner},
        {"Key": 'Group', "Value": GROUP_NAME},
        {"Key": CLUSTER_NAME_KEY, "Value": cluster_name},
        {"Key": "storage_name", "Value": storage_name}
    ]
    head_tags = common_tags + [
        {"Key": 'Name', "Value": 'HEAD_' + cluster_name},
        {"Key": ES_ROLE_KEY, "Value": "head"},
    ]
    worker_tags = common_tags + [
        {"Key": 'Name', "Value": 'WORK_' + cluster_name},
        {"Key": ES_ROLE_KEY, "Value": "worker"},
    ]
    return head_tags, worker_tags


def _compress_folder() -> str:
    """Compress the CWD to the parent/source.tar, return the repository name"""

    file_path = pathlib.Path.cwd().parent / TAR_NAME
    tar = tarfile.open(str(file_path), "w")

    # this string contained in the path? exclude
    excluded = ['loaded_from_sacred', '.git', '.idea', '.tar', '__pycache__', '.DS_Store', '.pytest_cache', 'blogpost']

    def filter_function(tarinfo):
        for ex in excluded:
            if ex in tarinfo.name:
                return None
        else:
            return tarinfo

    folder_name = pathlib.Path.cwd()

    print(f'Compressing {pathlib.Path.cwd()} to {file_path} ')
    tar.add(folder_name, recursive=True, filter=filter_function, arcname=folder_name.parts[-1])
    tar.close()
    return folder_name.stem


@cli.command()
def compress():
    path = _compress_folder()
    print(f'\n Done, source stored to : {path}')


def conf_redis() -> str:
    return f'echo "unixsocket {REMOTE_MASTER_SOCKET_PATH}" >> /etc/redis/redis.conf'


def make_master_script(download_code: str, run_command: str, repo_name: str):
    config_redis = """#!/bin/bash

# prepare the log file    
rm /home/ubuntu/user_data.log
touch /home/ubuntu/user_data.log
chown ubuntu /home/ubuntu/user_data.log

{
set -x

%s

# Disable redis snapshots
echo 'save ""' >> /etc/redis/redis.conf

# Make the unix domain socket available for the master client
# (TCP is still enabled for workers/relays)
%s
echo "unixsocketperm 777" >> /etc/redis/redis.conf
# allow remote access on the redis server (TODO should be more secure here)
echo "bind 0.0.0.0" >> /etc/redis/redis.conf

mkdir -p /var/run/redis
chown ubuntu:ubuntu /var/run/redis
systemctl restart redis

%s

} >> /home/ubuntu/user_data.log 2>&1

%s
    """ % (download_code, conf_redis(), set_python_path(repo_name), run_command)
    return config_redis


def set_python_path(repo_name: str) -> str:
    return f'cd /home/ubuntu/{repo_name}/ && export PYTHONPATH=.'


def make_master_run_script(run_command: str) -> str:
    return f'sudo -H -u ubuntu bash -c ' + \
           f'\'source /home/ubuntu/.bashrc && export PYTHONPATH=. && ' + \
           run_command + \
           f'>> /home/ubuntu/user_data.log 2>&1\''


def make_worker_script(download_code: str, run_command: str, repo_name: str):
    config_redis = """#!/bin/bash

rm /home/ubuntu/user_data.log
touch /home/ubuntu/user_data.log
chown ubuntu /home/ubuntu/user_data.log

{
set -x

%s

# Disable redis snapshots
echo 'save ""' >> /etc/redis/redis.conf

# Make redis use a unix domain socket and disable TCP sockets
sed -ie "s/port 6379/port 0/" /etc/redis/redis.conf
%s
echo "unixsocketperm 777" >> /etc/redis/redis.conf
mkdir -p /var/run/redis
chown ubuntu:ubuntu /var/run/redis

systemctl restart redis

%s

} >> /home/ubuntu/user_data.log 2>&1

%s
    """ % (download_code, conf_redis(), set_python_path(repo_name), run_command)
    # return base64.b64encode(config_redis)
    return config_redis


def make_worker_run_script(master_private_ip: str, run_command: str):
    """ Adds the correct --master_host and --relay_socket_path automatically here
    """
    return (f'sudo -H -u ubuntu bash -c '
            f'\'source /home/ubuntu/.bashrc && export PYTHONPATH=. && '
            'MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '
            f'{run_command} '
            f'--master_host {master_private_ip} '
            f'--relay_socket_path {REMOTE_MASTER_SOCKET_PATH} '
            f'>> /home/ubuntu/user_data.log 2>&1\'')


@cli.command()
@click.option('--key_name', default=AWS_KEY_NAME, help='AWS access key name')
@click.option('--cluster_name', type=str, default=None, help='cluster name, randomly chosen word by default')
@click.option('--size', type=int, default=1, help='Num workers to run')
@click.option('--master_type', default='t3.xlarge', help='Master instance type?')  # 2GB RAM might not be enough
@click.option('--worker_type', default='t3.2xlarge', help='Worker instance type?')
@click.option('--owner', default=OWNER, help='Owner of the machines')
@click.option('--image_id', default=DEFAULT_IMAGE_ID, help='image used for master and workers')
@click.option('--bucket_name', default=DEFAULT_BUCKET_NAME, help='name of the s3 storage for the code')
@click.option('--worker_command', default='python -u es/worker.py workers',
              help='python script to be ran on the worker machine, adds master_host and relay_socket_path..')
@click.option('--workers_per_machine', default=1, help='how many workers to run per one worker instance')
@click.option('--config', default='lander', help='configuration of the experiment to run on the head')
def launch(
        key_name: str,
        size: int,
        master_type: str,
        worker_type: str,
        image_id: str,
        owner: str,
        bucket_name: str,
        worker_command: str,
        config: str,
        cluster_name: Optional[str],
        workers_per_machine: int
):
    """Launches a cluster with head and N workers, all in an on-demand setting. See manage.py / set_price for spot."""

    if cluster_name is None:
        # credit for the words_alpha.txt file https://github.com/dwyl/english-words
        cluster_name = random.choice([word for word in open("words_alpha.txt")])[:-1]
    storage_name = cluster_name + '_' + datetime.now().strftime('%Y%m%d%H%M%S')  # name of the file storage on s3
    head_tags, worker_tags = get_tags(owner, cluster_name, storage_name)  # tags for head and workers

    print(f'Launching cluster named ------------ {cluster_name} --------------------- (storage_name: {storage_name})')
    print(f'---------------------------------------------------------------------------------------------------')

    ec2 = boto3.resource("ec2")
    as_client = boto3.client('autoscaling')

    # compress and upload the source code to the s3
    repo_name = _compress_folder()
    filename = str(pathlib.Path.cwd().parent / TAR_NAME)
    print(f'Uploading {filename} to {storage_name}')
    up(bucket_name, storage_name, filename)
    # down(bucket_name, storage_name, filename)  # just to check file available
    print(f'Upload finished')

    download_untar = f'rm -f /home/ubuntu/{TAR_NAME} && ' \
                     f'aws s3 cp s3://{bucket_name}/{storage_name} /home/ubuntu/{TAR_NAME} && ' + \
                     f'rm -rf /home/ubuntu/{repo_name} && ' + \
                     f'mkdir /home/ubuntu/{repo_name} && ' + \
                     f'tar -xvf /home/ubuntu/{TAR_NAME} -C /home/ubuntu/'

    head_command = 'python -u es/experiment.py with ' + config + ' local=False'
    master_script = make_master_script(download_untar, make_master_run_script(head_command), repo_name)

    print(f'master will run this: -------\n{master_script}\n--------------')

    master_instance = ec2.create_instances(
        ImageId=image_id,
        KeyName=key_name,
        InstanceType=master_type,
        MinCount=1,
        MaxCount=1,
        SecurityGroupIds=[DEFAULT_SECURITY_GROUP],
        UserData=master_script,
        # https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/user-data.html
        TagSpecifications=[{'ResourceType': 'instance', 'Tags': head_tags}],
        IamInstanceProfile={'Name': 'redis_cluster_code_access'},
        # EbsOptimized=True,
        # Tags=head_tags
    )[0]

    master_ip = master_instance.private_ip_address

    print(f'Master launched, IP is: {master_ip}')
    scaling_client = boto3.client("autoscaling")

    # try deleting the auto-scaling group and launch configuration of given name (should be done in the manage/kill)
    try:
        _ = scaling_client.delete_auto_scaling_group(
            AutoScalingGroupName=cluster_name,
            ForceDelete=True,
        )
        print(f'Auto scaling group named {cluster_name} deleted')
        # time.sleep(1)
    except:
        print(f'auto scaling group not found, skipping deletion')
    try:
        _ = scaling_client.delete_launch_configuration(
            LaunchConfigurationName=cluster_name
        )
        # time.sleep(1)
        print(f'Launch fonfig named {cluster_name} deleted')
    except:
        print(f'launch config not found, not deleting')

    worker_command = worker_command + f' --num_workers={workers_per_machine}'
    worker_script = make_worker_script(download_untar, make_worker_run_script(master_ip, worker_command), repo_name)
    print(f'Worker will run this: -------\n{worker_script}\n--------------')
    print(f'Creating launch configuration..')

    config_resp = as_client.create_launch_configuration(
        ImageId=image_id,
        KeyName=key_name,
        InstanceType=worker_type,
        LaunchConfigurationName=cluster_name,
        SecurityGroups=[DEFAULT_SECURITY_GROUP],
        UserData=worker_script,
        IamInstanceProfile=REDIS_CLUSTER_CODE_ACCESS,
        # EbsOptimized=True,
    )
    assert config_resp["ResponseMetadata"]["HTTPStatusCode"] == 200

    print(f'Creating auto scaling group..')

    asg_resp = as_client.create_auto_scaling_group(
        AutoScalingGroupName=cluster_name,
        LaunchConfigurationName=cluster_name,
        MinSize=size,
        MaxSize=size,
        DesiredCapacity=size,
        AvailabilityZones=AVAILABILITY_ZONES,
        Tags=worker_tags,
    )
    assert asg_resp["ResponseMetadata"]["HTTPStatusCode"] == 200

    print(f'\nCluster created, name: {cluster_name}\n')


if __name__ == '__main__':
    cli()
