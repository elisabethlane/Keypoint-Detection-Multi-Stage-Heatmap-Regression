import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras import optimizers

class Model():
    
    def __init__(self, INPUT_SHAPE, NUM_KEYPOINTS):
        
        self.input_shape = INPUT_SHAPE
        self.num_kp = NUM_KEYPOINTS
        self.model = self.build_model()
        self.loss_function = self.get_loss_func()
        self.optimiser = optimizers.Adam(learning_rate=1e-5)
        
        self.model.compile(loss=self.loss_function,
                            optimizer=self.optimiser)
        
        #print(self.model.summary())


    # Create a single stage FCN
    def build_model(self):
        
        outputs = []
      
        img = Input(self.input_shape, name="Input_stage")
      
        ### Stage 1 ###
        heatmaps1 = self.stages(img, 1, self.num_kp)
        outputs.append(heatmaps1)
      
        ### Stage 2 ###
        x = Concatenate()([img, heatmaps1])
        heatmaps2 = self.stages(x, 2, self.num_kp) 
        outputs.append(heatmaps2)
      
        model = keras.models.Model(inputs=img, outputs=outputs, name="FCN_Final")
        return model
    
    
        # Helper function for building model
    def conv_block(self, x, nconvs, n_filters, block_name, wd=None):
        for i in range(nconvs):
            x = Conv2D(n_filters, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                       kernel_regularizer=wd, name=block_name + "_conv" + str(i + 1))(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', name=block_name + "_pool")(x)
        return x
  
    
    # Represents one stage of the model
    def stages(self, x, stage_num, num_keypoints):
      
        #Block 1
        x = self.conv_block(x, nconvs=2, n_filters=64, block_name="block1_stage{}".format(stage_num))
      
        #Block 2
        x = self.conv_block(x, nconvs=2, n_filters=128, block_name="block2_stage{}".format(stage_num))

        #Block 3
        pool3 = self.conv_block(x, nconvs=3, n_filters=256, block_name="block3_stage{}".format(stage_num))

        #Block 4
        pool4 = self.conv_block(pool3, nconvs=3, n_filters=512, block_name="block4_stage{}".format(stage_num))
      
        #Block 5
        x = self.conv_block(pool4, nconvs=3, n_filters=512, block_name="block5_stage{}".format(stage_num))

        #Convolution 6
        x = Conv2D(4096, kernel_size=(1, 1), strides=1, padding="same", activation="relu", name="conv6_stage{}".format(stage_num))(x)
  
        #Convolution 7
        x = Conv2D(15, kernel_size=(1, 1), strides=1, padding="same", activation="relu", name="conv7_stage{}".format(stage_num))(x)

        #upsampling
        preds_pool3 = Conv2D(15, kernel_size=(1, 1), strides=1, padding="same", name="preds_pool3_stage{}".format(stage_num))(pool3)

        preds_pool4 = Conv2D(15, kernel_size=(1, 1), strides=1, padding="same", name="preds_pool4_stage{}".format(stage_num))(pool4)

        up_pool4 = Conv2DTranspose(filters=15, kernel_size=2, strides=2, activation='relu', name="ConvT_pool4_stage{}".format(stage_num))(preds_pool4)
    
        up_conv7 = Conv2DTranspose(filters=15, kernel_size=4, strides=4, activation='relu', name="ConvT_conv7_stage{}".format(stage_num))(x)

        fusion = Add()([preds_pool3, up_pool4, up_conv7])
      
        heatmaps = Conv2DTranspose(filters=15, kernel_size=8, strides=8, activation='relu', name="convT_fusion_stage{}".format(stage_num))(fusion)

        heatmaps = Conv2D(num_keypoints, kernel_size=(1, 1), strides=1, padding="same", activation="linear", name="output_stage{}".format(stage_num))(heatmaps)
    
        return heatmaps

 
    
    #Training the model using mean squared losss
    def get_loss_func(self):
        
        def mse(x, y):
            return mean_squared_error(x,y)
      
        keys = ['output_stage1', 'output_stage2']
        losses = dict.fromkeys(keys, mse)
        
        return losses

