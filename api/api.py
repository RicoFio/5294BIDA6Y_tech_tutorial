import time

from flask import (
    Flask,
    request,
    jsonify,
)
from flask_caching import Cache
from flask_swagger import swagger
from flask_swagger_ui import get_swaggerui_blueprint
import uuid
from scipy import stats

config = {
    "DEBUG": True,
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 3600  # one hour
}
app = Flask(__name__)

app.config.from_mapping(config)
cache = Cache(app)

# Swagger specific
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "LinReg API"
    }
)
app.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)


@app.route("/", methods=['GET'])
def specs():
    swag = swagger(app)
    swag['info']['version'] = "1.0"
    swag['info']['title'] = "Linear Prediction API"
    return jsonify(swag)


@app.route("/fit", methods=['POST'])
def linear_fit():
    """
    Fit a new model given some points
    ---
    tags:
      - model fitting
    parameters:
      - in: body
        name: body
        schema:
          id: Model
          required:
            - xs
            - ys
          properties:
            xs:
              type: List[float]
              description: list of floats
            ys:
              type: List[float]
              description: list of floats
    responses:
      201:
        description: Prediction created
      403:
        description: Bad request; The provided data cannot be parsed
      500:
        description: Internal server error
    """
    points_to_fit = request.get_json()
    # Check if the required data is in the request payload
    if not all([key not in points_to_fit.keys() for key in ['xs', 'ys']]):
        return {'error': 'either xs or ys are missing in the payload'}, 403
    # Check if xs and ys are of equal length
    if len(points_to_fit['xs']) != len(points_to_fit['ys']):
        return {'error': 'xs and ys need to be equal in length'}, 403
    # Now we can build the model
    model = stats.linregress(points_to_fit['xs'], points_to_fit['ys'])
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


@app.route("/predict/<model_id>", methods=['POST'])
def predict(model_id: str):
    # Check if model has been fitted before
    if not (model := cache.get(model_id)):
        return {'error': f'model_id {model_id} not found in cache. please fit your model first'}, 404
    else:
        data = request.get_json()
        # Check if required data is available
        if 'xs' not in data.keys():
            return {'error': 'xs are missing in the payload'}, 403
        # Make predictions
        predictions = model.intercept + model.slope * data['xs']
        response = {'ys': predictions}
        return response, 200
