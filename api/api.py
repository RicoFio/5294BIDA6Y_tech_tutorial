### IMPORTS ###
import os
import time
from typing import (
    List,
    Dict,
    Union,
)
import threading
import uuid

import numpy as np
from pydantic import (
    BaseModel,
    validator,
    Field,
)

from fastapi import (
    FastAPI,
    Response,
    status,
)
from fastapi.middleware.cors import CORSMiddleware

import diskcache as dc
from stressypy import create_job
###############

### FastAPI setup ###
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

cache = dc.FanoutCache('tmp')
#####################

WORKERS = int(os.getenv("WORKERS", 1))


class LinRegLocks:
    """
    A commodity class taking care to limit the number of concurrent
    operations to the number of workers available to the server
    """
    locks = {f'worker_{i}': threading.Lock() for i in range(WORKERS)}

    def __enter__(self):
        self.lock = self._get_lock()
        self.lock.acquire()

    def __exit__(self, *args, **kwargs):
        try:
            self.lock.release()
        except:
            pass

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
    """
    Pydantic definition of the data users can
    input to generate a fit together with
    the required validation
    """
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
    """Pydantic definition of the fitted model"""
    model_id: int
    model: Dict


class DataToPredict(BaseModel):
    """
    Pydantic definition of the data users can provide for inference
    """
    xs: List[float]


def linreg(x: np.array, y: np.array) -> Dict[str, float]:
    """
    The actual workhorse
    :returns
        dict with fitted slope and intercept
    """
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return {'slope': slope, 'intercept': intercept}


@app.post("/fit", status_code=status.HTTP_201_CREATED)
def linear_fit(points_to_fit: DataToFit,
               response: Response) -> Union[Dict[str, Union[str, Dict[str, float]]], Response]:
    """
    The endpoint to fit a line to a set of datapoints
    :param points_to_fit:
    :param response:
    :return:
    """
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
def predict(points_to_predict: DataToPredict, model_id: str):
    """
    The endpoint to predict the ys for the given xs given the
    previously fitted model
    :param points_to_predict:
    :param model_id:
    :return:
    """
    # Check if model has been fitted before
    if not (model := cache.get(model_id)):
        return {'error': f'model_id {model_id} not found in cache. please fit your model first'}, 404
    else:
        # Make predictions
        predictions = model['intercept'] + model['slope'] * np.array(points_to_predict.xs)
        response = {'ys': list(predictions)}

        return response
