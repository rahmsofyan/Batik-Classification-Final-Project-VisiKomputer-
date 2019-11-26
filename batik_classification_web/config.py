from os import getcwd
__ABS_PATH__ = getcwd()+"\\model\\"

DEBUG                =  False
MODEL_CLASSIFICATION = __ABS_PATH__+"VGG16_v1.h5"
MODEL_INFO           = __ABS_PATH__+"batik_info.json"
IP_ADDRESS           = "127.0.0.1"
PORT                 = 5000
