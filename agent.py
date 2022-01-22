from tensorflow.python.framework import ops
from pickle import TRUE
from sqlite3 import DatabaseError
import tensorflow as tf
from tensorflow.python.ops.nn_ops import dropout 
import numpy as np
import os 

from pathlib import Path



keepPb = 0.5
LSTMUnits=128
linearLayerUnits=512
PlaceCells_units=256
HeadCells_units=12

learning_rate=1e-5
clipping=1e-5
weightDecay=1e-5
batch_size=10
SGDSteps=300000
numberSteps=800
num_features=3 #num_features=[velocity, sin(angVel), cos(angVel)]

bins=32

#Number of trajectories to generate and use as data to train the network
num_trajectories=500

#Number of trajectories used to display the activity maps
showCellsTrajectories=5000

#Initialize place cells centers and head cells centers. Every time the program starts they are the same
rs=np.random.RandomState(seed=10)
#Generate 256 Place Cell centers
place_cell_centers=rs.uniform(0, 2.2 ,size=(PlaceCells_units,2))
#Generate 12 Head Direction Cell centers
head_cell_centers=rs.uniform(-np.pi,np.pi,size=(HeadCells_units))
global_step=0

#####################################################################################################################
"""
# Define Cutom  linear layer
class linear_Layer(tf.keras.layers.Layer):
    def __init__(self, units=512, input_dim=128):
        super(linear_Layer, self).__init__()
        self.w = self.add_weight(
            shape=(input_dim, units), initializer="glorot_normal", trainable=False
        )
        self.b = self.add_weight(shape=(units,), initializer="glorot_normal", trainable=False)

    def call(self, inputs):
        print("inputs: ", inputs)
        return tf.matmul(inputs, self.w) + self.b
"""
# Define Cutom  linear layer
class linear_Layer(tf.keras.layers.Layer):
    def __init__(self, units=512, input_dim=128):
        super(linear_Layer, self).__init__()
        self.LinearLayer_units = units    # 512
        self.linear_decoder_layer = tf.keras.layers.Dense(self.LinearLayer_units)
        self.linear_dropout_layer = tf.keras.layers.Dropout(0.5)

        # self.w = self.add_weight(
        #     shape=(input_dim, units), initializer="glorot_normal", trainable=False
        # )
        # self.b = self.add_weight(shape=(units,), initializer="glorot_normal", trainable=False)

    def call(self, inputs):        
        # return tf.matmul(inputs, self.w) + self.b
        x = self.linear_decoder_layer(inputs, training = True)
        # g_t_vector = self.linear_dropout_layer(x, training=True)
        return x


#####################################################################################################################
"""
# Define Custom output_placecells layer
class output_PlaceCells_layer(tf.keras.layers.Layer):
    def __init__(self, units=256, input_dim=512):
        super(output_PlaceCells_layer, self).__init__()
        self.w = self.add_weight(
            shape=(input_dim, units), initializer="glorot_normal", trainable=True
        )
        self.b = self.add_weight(shape=(units,), initializer="glorot_normal", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
"""

# Define Custom output_placecells layer
class output_PlaceCells_layer(tf.keras.layers.Layer):
    def __init__(self, units=256, input_dim=512):
        super(output_PlaceCells_layer, self).__init__()
        self.PlaceCells_units = units    # 256
        self.place_cells_layer = tf.keras.layers.Dense(self.PlaceCells_units)

        # self.w = self.add_weight(
        #     shape=(input_dim, units), initializer="glorot_normal", trainable=True
        # )
        # self.b = self.add_weight(shape=(units,), initializer="glorot_normal", trainable=True)

    def call(self, inputs):
        # return tf.matmul(inputs, self.w) + self.b
        y_t_vector = self.place_cells_layer(inputs)
        return y_t_vector

    #####################################################################################################################
# Define Custom output_headcells layer
class output_HeadCells_layer(tf.keras.layers.Layer):
    def __init__(self, units=12, input_dim=512):
        super(output_HeadCells_layer, self).__init__()
        self.HeadCells_units = units    # 12
        self.head_cells_layer = tf.keras.layers.Dense(self.HeadCells_units)

        # self.w = self.add_weight(
        #     shape=(input_dim, units), initializer="glorot_normal", trainable=True
        # )
        # self.b = self.add_weight(shape=(units,), initializer="glorot_normal", trainable=True)

    def call(self, inputs):
        # out_var = tf.matmul(inputs, self.w) + self.b
        # print("out_var: ", out_var)
        # return out_var
        # return tf.matmul(inputs, self.w) + self.b
        z_t_vector = self.head_cells_layer(inputs)
        return z_t_vector

#####################################################################################################################
# Define Custom output_headcells layer
class Place_HeadCells_layer(tf.keras.layers.Layer):
    def __init__(self, placeCellunits=256, headCellunits=12, hidden_unit=128):
        super(Place_HeadCells_layer, self).__init__()
        self.PlaceCells_units = placeCellunits    # 256
        self.Hidden_units = hidden_unit    # 128
        self.HeadCells_units = headCellunits      # 12

        # self.Wcp = self.add_weight(
        #     shape=(placeCellunits, hidden_unit), initializer="glorot_normal", trainable=False,name="Wcp"
        # )
        #
        # self.Wcd = self.add_weight(
        #     shape=(headCellunits, hidden_unit), initializer="glorot_normal", trainable=False,name="Wcd"
        # )

    def build(self, input_shape):
        # The __call__() method of your layer will automatically run build the first time it is called.
        self.Wcp = self.add_weight(
            shape=(self.PlaceCells_units, self.Hidden_units),
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
            name="Initial_state_cp",
        )
        self.Wcd = self.add_weight(
            shape=(self.HeadCells_units, self.Hidden_units),
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
            name="Initial_state_cd",
        )
        self.Whp = self.add_weight(
            shape=(self.PlaceCells_units, self.Hidden_units),
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
            name="Initial_state_hp",
        )
        self.Whd = self.add_weight(
            shape=(self.HeadCells_units, self.Hidden_units),
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
            name="Initial_state_hd",
        )

    def call(self, inputs):
        # placeCellinput = inputs[0]
        # headCellinput = inputs[1]
        # print("placeCellinput.shape, self.Wcp.shape ", placeCellinput.shape, self.Wcp.shape)
        # print("headCellinput.shape, self.Wcd.shape: ", headCellinput.shape, self.Wcd.shape)
        # print("placeCellinput.dtype, self.Wcp.dtype: ", placeCellinput.dtype, self.Wcp.dtype)
        # print("headCellinput.dtype, self.Wcd.dtype: ", headCellinput.dtype, self.Wcd.dtype)
        # return tf.matmul(placeCellinput, self.Wcp) + tf.matmul(headCellinput, self.Wcd)
        self.cell_state = tf.matmul(inputs[0], self.Wcp) + tf.matmul(inputs[1], self.Wcd)
        self.hidden_state = tf.matmul(inputs[0], self.Whp) + tf.matmul(inputs[1], self.Whd)
        return [self.hidden_state, self.cell_state]

#####################################################################################################################
loss_tracker = tf.keras.metrics.Mean(name="loss")
mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")


class CustomModel(tf.keras.Model):
    def train_step(self, data):
        X, Y = data        
        tf.config.run_functions_eagerly(True)
        
        with tf.GradientTape() as tape:
            y_pred1, y_pred2, y_pred_linearlayer = self(inputs=[X], training=True)            
            reshaped_label_place_cells = tf.reshape(Y[0], (-1, 256))            
            reshaped_label_head_cells = tf.reshape(Y[1], (-1, 12))
            self.errorPlaceCells = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=reshaped_label_place_cells, logits=y_pred1, name="Error_PlaceCells"))
            self.errorHeadCells = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=reshaped_label_head_cells, logits=y_pred2, name="Error_HeadCells"))
            self.weight_decay = 1e-5
            
            W3 = self.get_layer(index=8).trainable_weights[0]
            W2 = self.get_layer(index=7).trainable_weights[0]
                        
            l2_loss=self.weight_decay*tf.nn.l2_loss(W3) + self.weight_decay*tf.nn.l2_loss(W2)

            self.loss = self.errorHeadCells + self.errorPlaceCells + l2_loss
        # compute gradients
        trainable_vars = self.trainable_variables
        self.gvs = tape.gradient(self.loss, trainable_vars)

        self.clipping = 1        

        self.gvs[-4]=tf.clip_by_value(self.gvs[-4], -self.clipping, self.clipping)#, self.gvs[-4][1]]
        self.gvs[-3]=tf.clip_by_value(self.gvs[-3], -self.clipping, self.clipping)#, self.gvs[-3][1]]
        self.gvs[-2]=tf.clip_by_value(self.gvs[-2], -self.clipping, self.clipping)#, self.gvs[-2][1]]
        self.gvs[-1]=tf.clip_by_value(self.gvs[-1], -self.clipping, self.clipping)#, self.gvs[-1][1]]
                
        self.optimizer.apply_gradients(zip(self.gvs, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(self.loss)        
        mae_metric.update_state(Y[0], y_pred1)
        return {"loss": loss_tracker.result(), "mae": mae_metric.result()}

    @property
    def metrics(self):
        return [loss_tracker, mae_metric]

    def get_loss(self):
        loss_to_return = self.loss
        return loss_to_return
    
    def get_linearLayer(self):
        output_linearLayer = self.get_layer(index = 5).output # tf.keras.layers.Layer 's attribute 
        return output_linearLayer
    
    def get_layer_specified(self):
        layer = self.get_layer(index=8)
        return layer


#####################################################################################################################

#Define the agent structure network
class Network(tf.keras.layers.Layer):
    def __init__(self, lr, hu, lu, place_units, head_units, clipping, 
                 weightDecay, batch_size, num_features, n_steps ):        
        
        self.keepProb = 0.5

        self.epoch=tf.Variable(0, trainable=False)
        self.gvs = []
        #HYPERPARAMETERS
        self.learning_rate=lr
        self.Hidden_units=hu
        self.LinearLayer_units=lu
        self.PlaceCells_units=place_units
        self.HeadCells_units=head_units
        self.clipping=clipping
        self.weight_decay=tf.constant(weightDecay, dtype=tf.float32)
        self.batch_size=batch_size
        self.num_features=num_features
        self.step =0                

        self.buildNetwork()        
        #self.buildTensorBoardStats()        

        #self.saver=tf.train.Saver()
        self.file=tf.summary.create_file_writer("tensorboard/")     
        self.Loss=0
    
    def buildNetwork(self):

        self.placeCellGround=tf.keras.Input(shape=(self.PlaceCells_units),name="placecell")
        self.headCellGround=tf.keras.Input(dtype=tf.float32, shape=[self.HeadCells_units])

        #############################################################################################################
        
        self.LSTM_state = Place_HeadCells_layer(self.PlaceCells_units,
                                                self.HeadCells_units, 
                                                self.Hidden_units)([self.placeCellGround, self.headCellGround])
        
        input1 = tf.keras.Input(shape=(100, 3))
        lstm_output = tf.keras.layers.LSTM(128, return_sequences=True, return_state=False,
                                                             name="LSTM_layers")(input1, initial_state=self.LSTM_state)
        output_linear_layer = linear_Layer(self.LinearLayer_units,self.Hidden_units)(lstm_output)        
        linear_layerDrop = tf.keras.layers.Dropout(self.keepProb)(output_linear_layer)        
        output_placecells = output_PlaceCells_layer(self.PlaceCells_units, self.LinearLayer_units)(linear_layerDrop)               
        output_headcells = output_HeadCells_layer(self.HeadCells_units, self.LinearLayer_units)(linear_layerDrop)
       
        self.model = CustomModel(inputs = [input1, self.placeCellGround, self.headCellGround], outputs=[output_placecells, output_headcells, output_linear_layer])

        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(self.learning_rate, momentum=0.9), run_eagerly=True)
        
        self.LabelPlaceCells=tf.keras.Input(dtype=tf.float32, shape=[100, self.PlaceCells_units], name="Labels_Place_Cells")
        self.LabelHeadCells=tf.keras.Input(dtype=tf.float32,  shape=[100, self.HeadCells_units], name="Labels_Head_Cells")

        self.model.summary()             
        

    def train_network(self, X_data, placeCellGround_data, headCellGround_data, LabelPlaceCells_data, LabelHeadCells_data):

        self.model.fit(x =[X_data, placeCellGround_data, headCellGround_data],
                  y=[LabelPlaceCells_data, LabelHeadCells_data], epochs=1)            

        self.Loss = self.model.get_loss()
        

    
    def buildTensorBoardStats(self, willData,epoch, dataflag):
        #Episode data                
        with self.file.as_default(step=self.step):       
            if dataflag == 1:     
                tf.summary.scalar("mean_loss", willData)                
                tf.summary.scalar("epoch", epoch)
            if dataflag == 2:
                tf.summary.scalar("average_distance", willData)
                tf.summary.scalar("epoch", epoch)

        self.step +=1

    def save_restore_Model(self, restore, epoch=0):
        if restore:
            # self.saver.restore(self.sess, "agentBackup/graph.ckpt")
            path = os.getcwd()
            path = path+"\\agentBackup\\save_model"
            my_dir = Path(path)   
            self.model.load_weights(my_dir)
            print("restore_function")
        else:
            # self.sess.run(self.epoch.assign(epoch))
            # self.saver.save(self.sess, "agentBackup/graph.ckpt")
            path = os.getcwd()
            path = path+"\\agentBackup\\save_model"
            my_dir = Path(path)     
            print(my_dir)
            self.model.save_weights(my_dir)
            print("save_funciton")
   

















