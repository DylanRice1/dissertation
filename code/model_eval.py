from imports import *

print(tf.__version__)

model = load_model('InceptionV3.keras')

print(model.history.keys())