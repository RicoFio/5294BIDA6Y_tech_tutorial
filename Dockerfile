FROM python:3.8-slim AS linreg-api
WORKDIR /app
COPY ./api .
RUN apt-get update && \
  apt-get install -y --no-install-recommends gcc python3-dev libssl-dev stress && \
  pip install -r requirements.txt --no-cache-dir && \
  apt-get remove -y gcc python3-dev libssl-dev && \
  apt-get autoremove -y

ENV DEBUG 1

EXPOSE 8000

ENV WORKERS 1
ENV GUNICORN_CMD_ARGS "--workers ${WORKERS} --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000"

ENTRYPOINT ["gunicorn"]
CMD ["api:app"]