

CONTAINER_SIZE = 1000               #size of the container
OUTPUT_SIZE = 25                    #size of the image after reading and altering the original image
N = 1                               #Number of instances of brain images
N_CLASSES = 2                       #Number of different classifications
BLUR_COEF = 1.0 / 7.0               #Blurring Coefficient
IMAGE_SIZE = 25                     #This should be the same as output size
IMAGE_DEPTH = 25                    #This should be the same as the output size
IMAGE_CHANNELS = 1                  #Number of color channels
STEPS = 20000                       #Number of steps for training
BATCH_SIZE = 100                    #Batch size
LEARNING_RATE = 0.001               #Learning rate for training the CNN
OPTIMIZER = "SGD"                   #Loss function optimization algorithm. other options: {SGD,Adam,Adagrad,Ftrl,Momentum,RMSProp}
N_FEAT_LAYER1 = 32                  #Number of features output for conv layer 1
N_FEAT_LAYER2 = 64                  #Number of features output for conv layer 2
LOG_DIR = "../ops_logs"             #Directory where the logs would be stored for visualization of the training
DIRECTORY = "../TBI_data/"          #Directory where the input data is stored

