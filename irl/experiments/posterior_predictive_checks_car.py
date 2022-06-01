import random
from irl.irl_car import runFQI
random.seed(42)

runFQI(epoch=1000, evaluations=10, verbose=True)