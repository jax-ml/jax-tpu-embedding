# jax_tpu_embedding



To use this library, first create  a Google Cloud Account. You can, for example, refer to the `Set up and prepare a Google Cloud project` in [this guide](https://cloud.google.com/tpu/docs/v4-users-guide#project-setup).
Next, create a TPU VM by specifying your project ID and the accelerator type you want
(e.g. v2-8, v3-8, v4-8, v4-16, etc).

*Important:* for each accelerator type you need to choose the correct `version` and `zone`. For v4 the `version` is `tpu-vm-v4-base`, for v2 or v3 it is `tpu-vm-base`. For more details on zones, please refer to this [document](https://cloud.google.com/tpu/docs/regions-zones).

```
export TPU_NAME=your_tpu_name
export ZONE=us-central2-b
export RUNTIME_VERSION=tpu-vm-v4-base
export PROJECT_ID=your_project_id
export ACCELERATOR_TYPE=v4-8

gcloud compute tpus tpu-vm create ${TPU_NAME} --zone ${ZONE} --accelerator-type ${ACCELERATOR_TYPE} --version ${RUNTIME_VERSION} --project ${PROJECT_ID} --subnetwork=tpusubnet
```

If you are using single TPU device (e.g v4-8), then you can SSH to your VM and run the installations
directly from the VM. However, the  instructions below work for all sizes unless otherwise mentioned.

**Step 1.** Install JAX and flax by running  the following command:

```
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all --command="pip install --upgrade 'jax[tpu]==0.4.4' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all --command="pip install flax==0.6.7"
```

You can validate that the installation works properly by running the following command:

```
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all --command="TPU_PLATFORMS=tpu,cpu python3 -c 'import jax; print(jax.device_count())'"
```

**Step 2.** Install the following version of TensorFlow:

```
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --worker=all --zone=${ZONE}   --command="gsutil cp gs://cloud-tpu-tpuvm-artifacts/tensorflow/20230214/tf_nightly-2.13.0-cp38-cp38-linux_x86_64.whl .
pip install tf_nightly-2.13.0-cp38-cp38-linux_x86_64.whl"
```

**Step 3.** Clone this repository to all workers.

```
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --worker=all  --zone ${ZONE}  --command="git clone https://github.com/jax-ml/jax-tpu-embedding.git" 
```

##### Run some examples:
Run a script that works both on multihost and single host settings. The example uses pmap.

```
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all --command="source ~/jax-tpu-embedding/jax_tpu_embedding/scripts/export_tpu_pod_env.sh && PYTHONPATH=$PYTHONPATH:~/jax-tpu-embedding python3 ~/jax-tpu-embedding/jax_tpu_embedding/examples/pmap_example.py"
```

Examples with pjit that only run on a single host (e.g. v4-8) are shown below. In future, we will provide examples with pjit that run on multihost as well. This example can be run as distributed training by either replicating the model's dense layers, or by sharding them. To test the first case, simply run: 

```
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all --command="source ~/jax-tpu-embedding/jax_tpu_embedding/scripts/export_tpu_pod_env.sh && PYTHONPATH=$PYTHONPATH:~/jax-tpu-embedding python3 ~/jax-tpu-embedding/jax_tpu_embedding/examples/singlehost_pjit_example.py"
```
To shard your model across devices, you can pass the `--is-replicated=false` flag to the command as follows: 

```
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all --command="source ~/jax-tpu-embedding/jax_tpu_embedding/scripts/export_tpu_pod_env.sh && PYTHONPATH=$PYTHONPATH:~/jax-tpu-embedding python3 ~/jax-tpu-embedding/jax_tpu_embedding/examples/singlehost_pjit_example.py --is-replicated=false"
```


Note: The previous examples for a single host can also be run by sshing to the worker (i.e., the TPU VM) and running the script directly from the VM.

```
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} 

(vm) source ~/jax-tpu-embedding/jax_tpu_embedding/scripts/export_tpu_pod_env.sh && PYTHONPATH=$PYTHONPATH:~/jax-tpu-embedding python3 ~/jax-tpu-embedding/jax_tpu_embedding/examples/singlehost_pjit_model_parallel.py
```

## Running a jupyter notebooks on Cloud TPUs!
For your convenience two notebook examples are provided that run on single host (e.g v4-8). To test them, first SSH to the TPU VM by port-forwarding. n this example, that is port 8080.

```
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone=us-central2-b    --ssh-flag="-4 -L 8080:localhost:8080"
```
Run the following installations on the VM. Export the necessary environment variables and launch your notebook:

```
pip install jupyterlab
pip install markupsafe==2.0.1

export PYTHONPATH=$PYTHONPATH:~/jax-tpu-embedding
export TPU_NAME=local
jupyter notebook --no-browser --port=8080
```

Open `http://localhost:8080` on your local machine to see the repo and the examples.


