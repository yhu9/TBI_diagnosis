

CONTAINER_SIZE = 1000               #size of the container
OUTPUT_SIZE = 20                    #size of the image after reading and altering the original image
N = 1                               #Number of instances of brain images
N_CLASSES = 2                       #Number of different classifications
BLUR_COEF = 1.0 / 7.0               #Blurring Coefficient
IMAGE_SIZE = 20                     #This should be the same as output size
IMAGE_DEPTH = 20                    #This should be the same as the output size
IMAGE_CHANNELS = 1                  #Number of color channels
STEPS = 10                      #Number of steps for training
BATCH_SIZE = 2                      #Batch size
LEARNING_RATE = 0.01               #Learning rate for training the CNN
OPTIMIZER = "SGD"                   #Loss function optimization algorithm. other options: {SGD,Adam,Adagrad,Ftrl,Momentum,RMSProp}
N_FEAT_LAYER1 = 32                  #Number of features output for conv layer 1
N_FEAT_LAYER2 = 64                  #Number of features output for conv layer 2
N_FEAT_FULL1 = 1024                 #Number of features output for fully connected layer1
KEEP_RATE = 0.8                     #Rate of dropping out in the dropout layer
LOG_DIR = "../ops_logs"             #Directory where the logs would be stored for visualization of the training
DIRECTORY = "../test_data/"         #Directory where the input data is stored
STEPS_RECORD = 2                   #Number of steps to record data
