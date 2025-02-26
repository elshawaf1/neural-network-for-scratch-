'''Write the python code for the previous network for the tanh done 
activation function.
• choose the weight random from interval [-0.5, 0.5]. done 
• b1, b2= 0.5, 0.7 respectively. done 
• Print the output of the network. done
'''
import numpy as np

def tanh_activation_function (x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

w1= np.random.uniform(-0.5, 0.5, size=2)  
w2 = np.random.uniform(-0.5, 0.5) 

x = np.array([0.1, 0.2])
b1 = 0.5  
b2 = 0.7  
z1 = np.dot(w1, x) + b1 # frist 
h = tanh_activation_function(z1)  # after activation function 

z_output = w2 * h + b2 
o = tanh_activation_function(z_output)      

print(f"output {o}")