import os
import time

from fastapi import FastAPI
from pydantic import BaseModel, validator
from typing import List, Dict
import diskcache as dc
import numpy as np

import uuid
from scipy import stats

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
cache = dc.Cache('tmp')


class DataToFit(BaseModel):
    xs: List[float]
    ys: List[float]

    @validator('xs')
    def points_must_be_of_same_size(cls, v, values, **kwargs):
        if 'xs' in values and len(v) != len(values['ys']):
            raise ValueError('xs and ys have to be of same size')
        return v


class DataFittedModel(BaseModel):
    model_id: int
    model: Dict


class DataToPredict(BaseModel):
    xs: List[float]


@app.post("/fit")
def linear_fit(points_to_fit: DataToFit):
    # Check if xs and ys are of equal length
    if len(points_to_fit.xs) != len(points_to_fit.ys):
        return {'error': 'xs and ys need to be equal in length'}, 403
    # Now we can build the model
    model = stats.linregress(points_to_fit.xs, points_to_fit.ys)
    # Create a pseudo-random ID for it
    model_id = str(uuid.uuid4())
    # Store it temporarily
    cache.set(model_id, model)

    # Simulate that this takes A LONG TIME
    time.sleep(20)

    # Return the model id and its parameters
    response = {
        'model_id': model_id,
        'model': model
    }
    return response, 201


@app.post("/predict/{model_id}")
def predict(points_to_fit: DataToPredict, model_id: str):
    # Check if model has been fitted before
    if not (model := cache.get(model_id)):
        return {'error': f'model_id {model_id} not found in cache. please fit your model first'}, 404
    else:
        # Make predictions
        predictions = model.intercept + model.slope * np.array(points_to_fit.xs)
        response = {'ys': list(predictions)}

        return response, 200
