'''
A quick little guide on how to check if the machine has available GPU's for training and how to check that your model is using them.

'nvidia-smi' : A command that checks what available GPU's are in the system.

'nvidia-smi -l' : Same command that continuously monitors GPU usages throughout training

'''

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))