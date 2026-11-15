from fastapi.testclient import TestClient
import pytest
#import the application object defined in main.py
from main import create_app 

# create reusable test client for all the functions
# client = TestClient(create)


class fake_iris_model:
    def predict(self, features):
        return [0]

def fake_model_loader():
    return fake_iris_model()

def fake_version_loader():
    return "test_version"

@pytest.fixture
def client():
  app = create_app(
      model_loader=fake_model_loader,
      version_loader=fake_version_loader,
  )
  with TestClient(app) as  test_client:
      yield test_client

# we create fake model because we need to test api keys not model but now it depends on model-flow so we create fake-model


def test_health_endpoint(client):
    
    '''
      Verify that the application is running and can respond to health check request
    assert is like if else if code correct then good if not then assertion error
    '''
    response = client.get("/health")

    assert response.status_code == 200
    #checking response.body
    assert response.json()["status"] == "healthy"
    assert response.json()["model_version"] == "test_version"

def test_valid_iris_prediction(client):

    '''
      verify that valid flower measurements produce a prediction
    '''
    payload = {
  "sepal_length": 1,
  "sepal_width": 1,
  "petal_length": 1,
  "petal_width": 1
    }
    #send simulated HTTP post request with JSON input 

    response = client.post("/predict", json = payload)

    assert response.status_code == 200

    #checking response.body

    # convert the json response into a python dictionary 
    result = response.json()
    
    # then check contracts : both fields must exists
    assert "prediction" in result
    assert "class_name" in result
    
    #predicted class must be one of three iris class_names = ["setosa", "versicolor", "virginica"]
    assert result["class_name"] in ["setosa", "versicolor", "virginica"]
    #prediction must be 0,1,2
    assert result["prediction"] in [0,1,2]

def test_iris_input(client):
    """
    verify that invalid inputs are reject by fast api

    """
    payload = {
      "sepal_length": -1,
      "sepal_width": 1,
      "petal_length": 1,
      "petal_width": 1
        }
        #send simulated HTTP post request with JSON input 
    
    response = client.post("/predict", json = payload)

    # FastAPI normally returns HTTP 422 for validation error
    
    assert response.status_code == 422
  