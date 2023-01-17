ARG image_name
ARG base_image="gcr.io/jax-tpu-embeddings/${image_name}:latest"
FROM $base_image
RUN rm -rf /jax_tpu_embedding/jax_tpu_embedding
COPY . /jax_tpu_embedding_new
RUN git clone https://github.com/jax-ml/jax-tpu-embedding.git
RUN mv /jax_tpu_embedding_new/jax_tpu_embedding /jax_tpu_embedding/
RUN pip install /jax_tpu_embedding/jax_tpu_embedding/pip_package
RUN cd /jax_tpu_embedding && bazel build ...
WORKDIR /
CMD ["/bin/bash"]
