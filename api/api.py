import os
import time

from fastapi import (
    FastAPI,
    Response,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import (
    BaseModel,
    validator,
    Field,
)
from typing import List, Dict
import diskcache as dc
import numpy as np
import threading
from stressypy import create_job

import uuid

DEBUG = os.environ['DEBUG'] if os.environ['DEBUG'] else True

config = {
    "DEBUG": DEBUG,
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 3600  # one hour
}

app = FastAPI(
    debug=DEBUG,
    title='LinReg API',
    description='An amazing API for some OP linear regression',
    version='0.0.1',
    docs_url='/',
)

origins = [
    "http://localhost:8000",
    "http://0.0.0.0:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cache = dc.Cache('tmp')

WORKERS = int(os.getenv("WORKERS", 1))


class LinRegLocks:
    locks = {f'worker_{i}': threading.Lock() for i in range(WORKERS)}

    def __enter__(self):
        self.lock = self._get_lock()
        self.lock.acquire()

    def __exit__(self, *args, **kwargs):
        self.lock.release()

    def _get_lock(self):
        while self._all_locked():
            time.sleep(1)

        for lock in self.locks.values():
            if not lock.locked():
                return lock

    def _all_locked(self):
        return all([lock.locked() for lock in self.locks.values()])


linreg_lock = LinRegLocks()


class DataToFit(BaseModel):
    xs: List[float] = Field(example=[1, 2, 3])
    ys: List[float] = Field(example=[1, 2, 3])

    @validator('xs')
    def points_must_be_of_same_size(cls, v, values, **kwargs):
        if 'xs' in values and len(v) != len(values['ys']):
            raise ValueError('xs and ys have to be of same size')
        return v

    @validator('xs')
    def points_must_be_at_least_two(cls, v, values, **kwargs):
        if 'xs' in values and len(v) < 2:
            raise ValueError('xs and ys have to be at least 2')
        return v


class DataFittedModel(BaseModel):
    model_id: int
    model: Dict


class DataToPredict(BaseModel):
    xs: List[float]


def linreg(x: np.array, y: np.array):
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return {'slope': slope, 'intercept': intercept}


@app.post("/fit", status_code=status.HTTP_201_CREATED)
def linear_fit(points_to_fit: DataToFit, response: Response):
    # Check if xs and ys are of equal length
    if len(points_to_fit.xs) != len(points_to_fit.ys):
        return {'error': 'xs and ys need to be equal in length'}, 403

    # First check if all locks are already used up
    # If that's the case return 429
    if linreg_lock._all_locked():
        response.status_code = status.HTTP_429_TOO_MANY_REQUESTS
        return response

    # Now we can build the model
    # We use a thread lock to simulate a single threaded execution
    with linreg_lock:
        model = linreg(points_to_fit.xs, points_to_fit.ys)
        # Simulate that this takes A LOT of CPU for 20 seconds
        job = create_job(1, 20)
        job.run()

    # Create a pseudo-random ID for it
    model_id = str(uuid.uuid4())
    # Store it temporarily
    cache.set(model_id, model)

    # Return the model id and its parameters
    output = {
        'model_id': model_id,
        'model': model,
    }

    return output


@app.post("/predict/{model_id}", status_code=status.HTTP_200_OK)
def predict(points_to_fit: DataToPredict, model_id: str):
    # Check if model has been fitted before
    if not (model := cache.get(model_id)):
        return {'error': f'model_id {model_id} not found in cache. please fit your model first'}, 404
    else:
        # Make predictions
        predictions = model['intercept'] + model['slope'] * np.array(points_to_fit.xs)
        response = {'ys': list(predictions)}

        return response
