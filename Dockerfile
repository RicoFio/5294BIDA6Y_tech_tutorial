FROM mambaorg/micromamba:0.21.2 AS linreg-api
WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER api .
RUN micromamba install -y -f ./environment.yml &&\
    micromamba clean --all --yes

# Make RUN commands use the new environment:
SHELL ["/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "linreg-api", "/bin/bash", "-c"]

EXPOSE 8000

ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV DEBUG 1

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "gunicorn"]
CMD ["api:app", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]

