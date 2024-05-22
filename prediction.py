import numpy as np
from training_program import model
input_data= (eval(input("Input your data: ")))
input_data_as_numpy_array= np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction=model.predict(input_data_reshaped)

if prediction[0]== "R":
    print ("Object is a rock")
else:
    print("Object is a mine")