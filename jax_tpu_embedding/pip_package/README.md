# Steps to release a new pip package

Update the version number in setup.py and commit it.

In the following steps, we assume that the repo is stored at /tmp/jax_tpu_embedding.

After the docker file has been tested on a TPU VM, the following 
steps use the same Dockerfile and build/run docker on a cloudtop.
Build the docker image for building the pip package

```sh
docker build --tag jax:bc-pip - < pip_package/Dockerfile
```

Enter the docker image, mapping local directory /tmp/jax_tpu_embedding to docker directory /tmp/jax_tpu_embedding :

```sh
docker run --rm -it -v /tmp/jax_tpu_embedding:/tmp/jax_tpu_embedding --name <name> jax:bc-pip bash

#inside docker
cd /tmp/jax_tpu_embedding
```

From the /tmp/jax_tpu_embedding directory, run

```sh
rm -rf /tmp/wheels
PYTHON_MINOR_VERSION=8 pip_package/build.sh
```

If everything goes well, this will produce a wheel for python3.8 in
/tmp/wheels , you can copy to /tmp/jax_tpu_embedding/ and
test scp it to a VM

If this works successfully, you can then upload to the production server.
Remember to update the list of releases in the main README.