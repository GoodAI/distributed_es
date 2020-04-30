# Distributed Evolutionary Computation on RL Tasks

*By Jaroslav Vítků* \
*[GoodAI](https://www.goodai.com/)*

This code is based on OpenAI's ES (and reuses many of the original parts), it is modified on several places (see the [../README.md](../README.md)).

The repository is made to compare performance of OpenAI's ES with implementations of other evolutionary optimization methods in the [Nevergrad library](https://engineering.fb.com/ai-research/nevergrad/) on the stardardized RL task in OpenAI gym.


# Basic Requirements

All the requirements should be contained in the `requirements.txt`, furthermore,
 the python 3.7 (and 3.6 on AWS) and [PyTorch 1.5](https://pytorch.org/) (no GPU required) was used.

    pip install -r requirements.txt
   
Some dependencies need to be installed manually. For these, follow instruction how to install their dependencies the `requirements.txt`.

 
The experiment results are logged into the [Sacred](https://github.com/IDSIA/sacred) and observed using [Omniboard](https://github.com/vivekratnavel/omniboard) tool. 
These can be installed as a docker image:

1. Install [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/install/)
2. `cd sacred`
3. `docker-compose up -d`
4. Open `localhost:9000` in browser (note port 9000 can be changed in `sacred/docker-compose.yml`)
5. Uncheck any filters applied in the top row.
6. See the "Setting up the Sacred server" below for switching to local Sacred storage 


# Nevergrad

This describes how to run the experiments done using the Nevergrad library.
Note that Nevergrad experiments currently support single-machine setting,
 but could be easily configured to run on auto-scaled cluster if desired,
 since the workers are implemented as a [Ray workers](https://docs.ray.io/en/latest/autoscaling.html).

    
## Running the Experiment

To run the experiment, run the following from the root directory:

    python nevergrad_es/experiment.py with lander    

which will:

* load the configuration of lander defined in `nevergrad_es/experiment.py:lander()` named experiment config
* run the head which launches `num_processors` of workers and starts logging into the Sacred and console


## Observing the results

Now it should be possible to open the Omniboard address `localhost:9000` in the browser and observer the results.

It is possible to render a behavior of resulting policy in the gym environment even during the ES run. In order to do this, run:

    python env/render.py 1234
    
, where `1234` is experiment ID in the Omniboard.

This will:

* download the experiment configuration,
* instantiate the environment and the policy based on the configuration,
* download and deserialize the policy parameters from the last generation logged in `Omnobiard/artifacts`,
* run the rendering for given `num_episodes`


# OpenAI's ES - Local

There two possibilities how to run this algorithm, first the local version.
 Compared to the full distributed version (see the [../README.md](../README.md) schematics),
 the local one does not need the Local Relay and Redis local servers, since workers can interact directly with the main server.

Run the redis server:
    
    python es/head.py run-server --stop
    
Run workers:
    
    python es/worker.py start-workers --num_workers=7

Run the experiment which launches the head:
    
     python es/experiment.py with lander

This does the following:
 
 * launches the head which starts pushing tasks into the redis server
 * the workers pull the task from the server, compute fitness values and push them to the server
 * the head pops results from the server, until the `pop_size` of results is not received
 * the head logs into the console and Sacred.

Note that the head can launched/killed/restarted independently on the workers.
 If the worker detects the `Task` with different experiment start time, it re-initializes the experiment 
 (environment, policy, hyperparameters) from the configuration on the server and starts evaluating for the new experiment.

In order to switch workers to waiting mode after the head is killed, you can flush the redis database by calling:

    python es/head.py flush


# OpenAI's ES - Distributed

The distributed version is made to be ran on a cluster of AWS instances.
 There is 1 head and N worker instances in the cluster, each worker instance can run `workers_per_machine` ES workers.
 

## Setting up the AWS

In order to setup the AWS, the following steps have to be done.

* Setup your AWS account and:
    * write path to your `*.pem` file to the `es.launch.py:PEM_FILE`
    * write name of your AWS key to `es/launch.py:AWS_KEY_NAME`

### Setting up the Sacred server

For storing the results, there are two options: 

1. Each head instance could have local Sacred database running. 
But this way the results have to be pulled from the head before terminating it. 
In this case, set the `utils/sacred_local.py:SACRED_LOCAL=True`.

2. It is better to setup shared Sacred database which keeps running (preferably) on AWS or locally.
 In this case, set the IP address of the machine running the Sacred server to 
 `badger_utils/sacred/sacred_config.py:SacredConfigFactory:mongo_url`.


### Preparing the pre-installed AMI

Prepare AMI with pre-installed dependencies for the cluster.
 These can be the same for head and workers 
(although the workers might benefit from more lightweight setup).

* Launch an AWS instance with Deep Learning support (e.g. `Deep Learning AMI (Ubuntu 18.04) Version 27.0`)
* Copy the source files to it using `python cluster/manage.py upload INSTANCE_IP`
* SSH into it, unpack the code and install all the requirements `mkdir distributed_es && tar -xvf /home/ubuntu/source.tar -C /home/ubuntu/ && cd distributed_es`
* Save as your AMI
* Read the AMI's ID and write it to the `es/launch.py:DEFAULT_IMAGE_ID` variable

### Preparing the S3 Storage

Launching the cluster proceeds as follows:

* Contents of this repository are packed and sent to the S3 storage
* Head and given number of worker instances is launched (each with specified AMI).
 In the `UserData` (startup commands) the instances contain address of this S3 storage. 
* On the startup, each instance downloads the code, unpacks it and runs either ES workers or the head's script 
(`python es/experiment.py with [--config]`).

Therefore these instances have to have access to the S3 storage.


#### Setting up the Access Role
 
First, it is good to setup restricted access to the S3 storage. This is configured as follows:

* Obtain your `UserId` by running in the terminal `aws iam get-user --user-name your.email@something.com`,
 read the field `UserId` and write it to `es/launch.py:DEFAULT_USER_ID`.
* Login to AWS console, create a new role under IAM/Roles in the console.
 Role can be called e.g. `redis_cluster_code_access` (or change also in `es/launch.py:REDIS_CLUSTER_CODE_ACCESS` variable) and configure it 
to enable the full access to S3, field called `AmazonS3FullAccess`.
* Obtain the `RoleId` code of `redis_cluster_code_access` by running the following from the terminal 
`aws iam get-role --role-name redis_cluster_code_access`, read the string under `RoleId` and store it to `es/launch.py:DEFAULT_ROLE_ID`.

Also set other unfilled variables on the top of the `es/launch.py`.


#### Creating the bucket

Now we can create bucket with the access restricted by these conditions.
 This is done just once, the bucket is reused for all clusters from that point.

* Create the bucket by calling `es/manage.py create-bucket` (if this fails, there is a name 
conflict and you have to modify the storage name and set this new name to `es/manage.py:DEFAULT_BUCKET_NAME`).
* This bucket denies all access by default, only user with the given `UserId` and `RoleId`
 and his instances can access it.


## Running the cluster

To run an experiment on a cluster, run the following:

    python cluster/launch.py launch --workers_per_machine=8 --size=5 --config='lander lr=0.01'
    
which runs the `lander` experiment with custom `lr` on a cluster with 5 worker instances (of default type),
 each running 8 ES workers. The cluster is automatically named using the [english-words](https://github.com/dwyl/english-words) repository.


## Managing the Cluster

To observe own currently running clusters, run in terminal:

     watch python cluster/manage.py list

which will show up-to-date list all heads and their workers owned by the `OWNER`.

To monitor console outputs of the cluster head, run:

    python cluster/manage.py tail CLUSTER_NAME

To terminate the cluster:

    python cluster/manage.py kill CLUSTER_NAME

Setting the fixed size with on-demand instances:

    python cluster/manage.py size CLUSTER_NAME

To resize or change configuration (e.g. to spot-instances) of the cluster, run the following for help. 
It is possible to switch from on-demand to spot instances (although there is a probably bug, see the limitations),
 change the size of the cluster, change the type of workers etc.

    python cluster/manage.py set-price --help

# Other 

To upload the repository code to a running instance (e.g. for running single-machine experiments with Nevergrad) run:

    python cluster/manage.py upload INSTANCE_IP
    
then follow the instructions in the terminal output of the script.


### Known limitations

* Simplifications of the code (this code is modified to support only `num_epsides` per fitness, the original code used more options to specify the runtime of evaluation)
* `gym.Env` instances not reused since the seeding does not work for some environments, on the other hand, some environments (halfCheetah and Walker) are memory leaking, so short epochs can (will) eat all memory, especially if the environment is restarted very often. The reuse of environment instances should be added (which breaks the determinism of evaluations)
* Automatic cluster shutdown would be useful, but not done for now. 
* `cluster/manage.py set-price` probably contains a bug which passes a wrong `UserData` to new workers, at least in case of custom `spot-price`

