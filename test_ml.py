import pytest
# TODO: add necessary import
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from ml.model import compute_model_metrics, train_model
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# TODO: implement the first test. Change the function name and input as needed
def test_one():
    """
    Test that the model created is a Random Forest Classifier
    """
    # Your code here
    X = np.array([[1,2],[2,3],[3,4],[4,5],[5,6]])
    y = np.array([1,2,3,4,5])
    
    model = train_model(X,y)
    
    assert isinstance(model, RandomForestClassifier), "The model is not a Random Forest Classifier"

# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    Check that fbeta, precision, and recall scores are within the proper range
    """
    y = np.array([1,1,0,1])
    preds = np.array([0,1,1,0])
    
    p, r, f = compute_model_metrics(y,preds)

    assert 1 >= p >= 0, "Precision is out of range"
    assert 1 >= r >= 0, "Recall is out of range"
    assert 1 >= f >= 0,  "Fbeta is out of range"


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    Test that training data is properly split
    """
    # Your code here
    project_path = os.getcwd()
    data_path = os.path.join(project_path, "data", "census.csv")
    print(data_path)
    data = pd.read_csv(data_path)

    train, test = train_test_split(data, test_size=.3, random_state=42)

    assert len(train) + len(test) == len(data), "Data is not properly split"