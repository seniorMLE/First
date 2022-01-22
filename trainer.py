from PIL.Image import merge
import numpy as np
from numpy import array
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

class Trainer():
    def __init__(self, agent, pcu, numSteps):
        self.agent=agent
        self.PlaceCells_units=pcu
        self.numberSteps=numSteps
        self.bool_met_criteria = False

    def training(self, X, Y, epoch, loss_criteria):
        
        mn_loss=0        
        
        #Divide the sequence in 100 steps in order to apply TBTT of 100 timesteps.
        for startB in range(0, self.numberSteps, 100):

            endB=startB+100

            #Retrieve the inputs for the 100 timesteps
            xBatch=X[:,startB:endB]
            
            #Retrieve the labels for the 100 timesteps
            yBatch=Y[:,startB:endB]

            #Retrieve label at timestep 0 for the 100 timesteps
            init_LSTM=yBatch[:,0]                        

            X_net = xBatch
            
            placeCellGround = init_LSTM[:, :self.PlaceCells_units]           
            headCellGround = init_LSTM[:, self.PlaceCells_units:]
            LabelPlaceCells = yBatch[:, :, : self.PlaceCells_units]
            LabelHeadCells = yBatch[:, :, self.PlaceCells_units:]
            # keepProb = 0.5
            self.agent.train_network(X_net, placeCellGround, headCellGround, LabelPlaceCells, LabelHeadCells)
                              
            self.meanLoss = self.agent.Loss            
            scalar_meanLoss = self.meanLoss.numpy()                
            mn_loss += scalar_meanLoss/(self.numberSteps//100)     

            # print("linearLayer_output :", self.agent.model.get_linearLayer())

            if self.meanLoss <= loss_criteria:
                self.bool_met_criteria = True
                break            

        #training epoch finished, save the errors for tensorboard               
        self.agent.buildTensorBoardStats(willData = mn_loss, epoch = epoch, dataflag = 1)
        return self.bool_met_criteria, self.meanLoss


    def testing(self, X, init_X, positions_array, pcc, epoch):
        avgDistance=0

        displayPredTrajectories=np.zeros((10,800,2))

        #Divide the sequence in 100 steps
        for startB in range(0, self.numberSteps, 100):
            
            endB=startB+100

            #Retrieve the inputs for the timestep
            xBatch=X[:,startB:endB]
            test_placeCellGroundData = init_X[:, (startB//100), :self.PlaceCells_units]
            test_headCellGroundData = init_X[:, (startB//100), self.PlaceCells_units:]         
            self.agent.keepProb = 0.8   
            
            test_outputModel = self.agent.model([xBatch, test_placeCellGroundData, test_headCellGroundData])
            
            placeCellLayer = test_outputModel[0].numpy()

            #retrieve the position in these 100 timesteps
            positions=positions_array[:,startB:endB]            

            #Retrieve which cell has been activated. Placecell has shape 1000,256. idx has shape 1000,1
            idx=np.argmax(placeCellLayer, axis=2)
            
            #Retrieve the place cell center of the activated place cell                                                
            predPositions=pcc[idx]

            #Update the predictedTrajectory.png
            if epoch%8000 == 0:
                displayPredTrajectories[:,startB:endB]=np.reshape(predPositions,(10,100,2))
            
            predPositions = np.reshape(predPositions,(-1,2))                

            #Compute the distance between truth position and place cell center
            distances=np.sqrt(np.sum((predPositions - np.reshape(positions, (-1,2)))**2, axis=1))
            avgDistance +=np.mean(distances)/(self.numberSteps//100)
        
        #testing epoch finished, save the accuracy for tensorboard
        
        self.agent.buildTensorBoardStats(willData = avgDistance, epoch = epoch, dataflag = 2)
        
        #Compare predicted trajectory with real trajectory
        print(epoch)
        if epoch%8000 == 0:
            rows=3
            cols=3
            fig=plt.figure(figsize=(40, 40))
            for i in range(rows*cols):
                ax=fig.add_subplot(rows, cols, i+1)
                #plot real trajectory.                

                plt.plot(positions_array[i,:,0], positions_array[i,:,1], 'b', label="Truth Path")
                #plot predicted trajectory
                plt.plot(displayPredTrajectories[i,:,0], displayPredTrajectories[i,:,1], 'go', label="Predicted Path")
                plt.legend()
                ax.set_xlim(0,2.2)
                ax.set_ylim(0,2.2)

            fig.savefig('predictedTrajectory.png')

