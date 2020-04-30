import json
import os
import pathlib
from random import random
from typing import List, Dict, Optional

import boto3
import click

from cluster.launch import ES_ROLE_KEY, CLUSTER_NAME_KEY, _compress_folder, TAR_NAME, OWNER, PEM_FILE, \
    DEFAULT_BUCKET_NAME, DEFAULT_USER_ID, DEFAULT_ROLE_ID


@click.group()
def cli():
    pass


@cli.command()
@click.option('--name', default=DEFAULT_BUCKET_NAME, help='name of the s3 storage for the code')
@click.option('--user_id', type=str, default=DEFAULT_USER_ID)
@click.option('--role_id', type=str, default=DEFAULT_ROLE_ID)
def create_bucket(name: str, user_id: str, role_id: str):
    """
    Creates bucket with access privileges only for the current AWS user (and his instances).

    - name: name of the bucket, do not change
    - ID of the user who launches the clusters, the access rights are passed to instances launched by the user
    """

    s3 = boto3.client('s3')
    session = boto3.session.Session()
    current_region = session.region_name

    print(f'---------------------------- region name is : {current_region}, creating bucket named: {name}')
    _ = s3.create_bucket(Bucket=name, CreateBucketConfiguration={'LocationConstraint': current_region})

    # this denies all access by default to the bucket, only exceptions are mentioned below
    args = {
        "Version": "2012-10-17",
        "Id": "VPCe and SourceIP",
        "Statement": [{
            "Sid": "VPCe and SourceIP",
            "Effect": "Deny",  # should deny all connections by default
            "Principal": "*",
            "Action": "s3:*",
            "Resource": [
                f'arn:aws:s3:::{name}',
                f'arn:aws:s3:::{name}/*'
            ],
            "Condition": {
                "StringNotLike": {
                    "aws:userId": [
                        # this is role redis_cluster_code_access (created using AWS console, grants reading from s3)
                        f'{role_id}:*',
                        # user to be allowed to connect (obtained from your login credentials)
                        f'{user_id}',
                        "111111111111"
                    ]
                }

            }
        }]
    }

    print(f'---------------------------- putting bucket policy')
    bucket_policy = json.dumps(args)
    s3.put_bucket_policy(Bucket=name, Policy=bucket_policy)

    print(f'---------------------------- reading bucket policy...(access test)')
    _ = s3.get_bucket_policy(Bucket=name)
    print(f'---------------------------  bucket is setup\n')


def _collect_instances(owner: str):
    """Collect instances (heads) that are running, have tag with key `ES_role` and belong to a given owner"""
    client = boto3.client("ec2")
    print("Collecting instances")
    instances = [x['Instances'][0] for x in client.describe_instances(
        Filters=[
            {
                'Name': f'tag:{ES_ROLE_KEY}',
                'Values': [
                    "head"
                ]
            },
            {
                'Name': 'instance-state-name',
                'Values': [
                    'running'
                ]
            },
            {
                'Name': 'tag:Owner',
                'Values': [
                    owner
                ]
            },
        ]
    )['Reservations']]
    return instances


def _collect_scaling_groups(owner: str) -> List:
    """Collect autoscaling groups that contain key `ES_role` and belong to the specified owner"""

    client = boto3.client("autoscaling")
    print("Collecting scaling groups")
    resp = client.describe_auto_scaling_groups()
    assert "NextToken" not in resp, "did not program to handle pagination"

    groups = resp['AutoScalingGroups']
    result = []
    for group in groups:
        if _get_tag_val(group['Tags'], 'Owner') == owner and \
                any([tag['Key'] == ES_ROLE_KEY for tag in group['Tags']]):
            result.append(group)
    return result


def _get_group_by_name(owner: str, name: str):
    groups = _collect_scaling_groups(owner)
    if len(groups) == 0:
        print(f'ERROR: no scaling group of a given owner not found!')
        return None
    for group in groups:
        if group['AutoScalingGroupName'] == name:
            return group

    print(f'ERROR: scaling group named {name} not found!')
    return None


def _get_tag_val(tags: List[Dict], key: str):
    """Get value of a tag of a given key"""
    for one_tag in tags:
        if one_tag['Key'] == key:
            return one_tag['Value']
    return ''


@cli.command()
@click.option('--owner', default=OWNER, help='Owner of the machines')
def list(owner: str):
    """Lists all the owner's clusters"""

    instances = _collect_instances(owner)
    groups = _collect_scaling_groups(owner)

    instance_names = [_get_tag_val(instance['Tags'], CLUSTER_NAME_KEY) for instance in instances]
    group_names = [group['AutoScalingGroupName'] for group in groups]

    ipKey = 'PrivateIpAddress'
    heads_formatted = [f'{name} (IP: {instance[ipKey]})' for name, instance in zip(instance_names, instances)]

    instKey = 'Instances'
    groups_formatted = [f'{name} ({len(group[instKey])} workers)' for name, group in zip(group_names, groups)]

    # print(f'\nfound {len(instances)} of running heads, names are: \n\t{instance_names}\n')
    print(f'Your running heads are: \n\n\t{heads_formatted}\n')
    print(f'Found groups: \n\n\t{groups_formatted}\n')


@cli.command()
@click.argument('cluster_name')
@click.option('--owner', default=OWNER, help='Owner of the machines')
def kill(owner: str, cluster_name: str):
    """Kills the cluster of a given name (searches only through clusters belonging to the owner)"""

    instances = _collect_instances(owner)
    groups = _collect_scaling_groups(owner)

    instances_by_name = []
    for instance in instances:
        if _get_tag_val(instance['Tags'], 'cluster_name') == cluster_name:
            instances_by_name.append(instance)

    groups_by_name = []
    for group in groups:
        if group['AutoScalingGroupName'] == cluster_name:
            groups_by_name.append(group)

    if len(instances_by_name) == 0 and len(groups_by_name) == 0:
        print(f'ERROR: no cluster with this name found!')
        return
    assert len(instances_by_name) == len(groups_by_name) == 1, \
        f'Expected just one instance and one group,' \
        f' but found {len(instances_by_name)} instances and {len(groups_by_name)} groups'

    print("This will kill the following cluster:")
    click.secho(f'\t\t\t\t\t{cluster_name}', fg="red")
    click.confirm('Continue?', abort=True)

    ec2_client = boto3.client("ec2")
    scaling_client = boto3.client('autoscaling')

    print(f'Deleting autoscaling group named {cluster_name}')
    _ = scaling_client.delete_auto_scaling_group(AutoScalingGroupName=cluster_name, ForceDelete=True)

    print(f'Deleting launch configuration named {cluster_name}')
    _ = scaling_client.delete_launch_configuration(LaunchConfigurationName=cluster_name)

    print(f'Terminating cluster head')
    ec2_client.terminate_instances(InstanceIds=[instances_by_name[0]["InstanceId"]])


@cli.command()
@click.argument('cluster_name')
@click.option('--price', type=str, default=None)
@click.option('--min_size', type=int, default=None)
@click.option('--max_size', type=int, default=None)
@click.option('--desired_size', type=int, default=None)
@click.option('--worker_type', default=None, help='Worker instance type')
@click.option('--owner', default=OWNER, help='Owner of the machines')
def set_price(
        price: Optional[str],
        min_size: int,
        max_size: int,
        desired_size: int,
        worker_type: str,
        owner: str,
        cluster_name: Optional[str]):
    """Spot instance config, set the price and target sizes of the cluster, change worker type"""
    group = _get_group_by_name(owner, cluster_name)
    if group is None:
        return

    min_size = group['MinSize'] if min_size is None else min_size
    max_size = group['MaxSize'] if max_size is None else max_size
    desired_size = group['DesiredCapacity'] if desired_size is None else desired_size

    print(f'This will change spot price in the group {cluster_name} to {price}$/hour')
    click.secho(f'\t\t\t\t\t{cluster_name}', fg="blue")
    click.confirm('Continue?', abort=True)

    as_client = boto3.client('autoscaling')
    configs = as_client.describe_launch_configurations(
        LaunchConfigurationNames=[cluster_name]
    )
    assert len(configs['LaunchConfigurations']) == 1, 'Unexpected number of launch configurations found'
    orig_config = configs['LaunchConfigurations'][0]
    instance_type = orig_config['InstanceType'] if worker_type is None else worker_type

    config = {'ImageId': orig_config['ImageId'],
              'KeyName': orig_config['KeyName'],
              'InstanceType': instance_type,
              'LaunchConfigurationName': orig_config['LaunchConfigurationName'],
              'SecurityGroups': orig_config['SecurityGroups'],
              'UserData': orig_config['UserData'],  # TODO possible problem with UserData in case of autoscaling
              'IamInstanceProfile': orig_config['IamInstanceProfile']}

    # set the spot price (or set it to on-demand instances)
    if price is not None:
        config['SpotPrice'] = price

    # clone the old launch config, set it to use by the group, and delete the old config
    temp_name = config['LaunchConfigurationName'] + f'_temp_{random()}'
    config['LaunchConfigurationName'] = temp_name
    config_resp = as_client.create_launch_configuration(**config)
    assert config_resp["ResponseMetadata"]["HTTPStatusCode"] == 200

    client = boto3.client("autoscaling")
    client.update_auto_scaling_group(
        AutoScalingGroupName=group["AutoScalingGroupName"],
        LaunchConfigurationName=temp_name)

    print(f'Deleting the old launch configuration')
    _ = as_client.delete_launch_configuration(LaunchConfigurationName=cluster_name)

    # crate new config with the original name (cluster_name) and point the group there
    print(f'Old launch config named deleted, creating new config named {cluster_name}')
    config['LaunchConfigurationName'] = cluster_name
    config_resp = as_client.create_launch_configuration(**config)
    assert config_resp["ResponseMetadata"]["HTTPStatusCode"] == 200

    client.update_auto_scaling_group(
        AutoScalingGroupName=group["AutoScalingGroupName"],
        LaunchConfigurationName=cluster_name,
        MinSize=min_size,
        MaxSize=max_size,
        DesiredCapacity=desired_size)

    _ = as_client.delete_launch_configuration(LaunchConfigurationName=temp_name)
    print(f'Spot price setup done!')


@cli.command()
@click.argument('cluster_name')
@click.argument('size')
@click.option('--owner', default=OWNER, help='Owner of the machines')
def resize(cluster_name: str, size: int, owner: str):
    """Change to fixed size: min_size=max_size=desired_capacity, expects the on-demand instances"""
    size = int(size)

    groups = _collect_scaling_groups(owner)
    if len(groups) == 0:
        print(f'ERROR: scaling group of the cluster named {cluster_name} not found!')
        return
    assert len(groups) == 1, 'found multiple scaling groups of this name'

    print(f'This will resize the scaling group to {size}')
    click.secho(f'\t\t\t\t\t{cluster_name}', fg="blue")
    click.confirm('Continue?', abort=True)

    group = groups[0]

    client = boto3.client(
        "autoscaling",
    )
    client.update_auto_scaling_group(
        AutoScalingGroupName=group["AutoScalingGroupName"],
        MinSize=size,
        MaxSize=size,
        DesiredCapacity=size,
    )
    print(f'\nResizing done!')


@cli.command()
@click.argument('cluster_name')
@click.option('--owner', default=OWNER, help='Owner of the machines')
@click.option('--key_location', default=PEM_FILE, help='location of AWS key')
def tail(cluster_name: str, owner: str, key_location: str):
    """Tails the cluster head output of the startup command"""
    instances = _collect_instances(owner)
    instances_by_name = []
    for instance in instances:
        if _get_tag_val(instance['Tags'], 'cluster_name') == cluster_name:
            instances_by_name.append(instance)

    if len(instances_by_name) == 0:
        print(f'ERROR: no cluster head with this name found: {cluster_name}')
        return
    assert len(instances_by_name) == 1, 'multiple heads with this name found!'

    instance = instances_by_name[0]
    head_ip = instance['PrivateIpAddress']

    print(f'Connecting to {cluster_name} with IP: {head_ip}')

    command = " ".join([
        "ssh",
        f'-i {key_location} '
        "ubuntu@" + head_ip,
        "'tail -f -n 2000 user_data.log && exec bash -l'"
    ])
    print(command)
    os.system(command)
    return


@cli.command()
@click.argument('ip', type=str)
@click.option('--pem_file',
              default=PEM_FILE,
              help='path to your *.pem file for accessing the machine')
@click.option('--dry_run/--no_dry_run', help='run without copying')
def upload(ip: str, pem_file: str, dry_run: bool):
    """Direct upload of contents of this repository to given machine (not for cluster)"""

    repo_name = _compress_folder()
    filename = str(pathlib.Path.cwd().parent / TAR_NAME)
    target_path = f'/home/ubuntu/'

    print(f'Uploading {filename} to ubuntu@{ip}:{target_path}')
    command = f'scp -i {pem_file} {filename} ubuntu@{ip}:{target_path}'
    if not dry_run:
        os.system(command)

    print('\nDONE, now ssh to the machine\n\n'
          f'ssh -i {pem_file} ubuntu@{ip}\n\n'
          'and untar the contents and prepare for run:\n\n'
          f'rm -rf {repo_name} && mkdir {repo_name} && tar -xvf /home/ubuntu/{TAR_NAME} -C /home/ubuntu/ && '
          f'cd {repo_name} && export PYTHONPATH=.\n')


if __name__ == '__main__':
    cli()
