import numpy as np
import os 
import matplotlib.pyplot as plt
import os
from scipy.signal import correlate2d
import tensorflow as tf

from ratSimulator import RatSimulator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def showGridCells(agent, dataGenerator, num_traj, num_steps, pcu, hcu, llu, bins, place_cell_centers, head_cell_centers):

    factor=2.2/bins
    activityMap=np.zeros((llu, bins, bins))
    counterActivityMap=np.zeros((llu, bins, bins))

    X=np.zeros((num_traj,num_steps,3))
    positions=np.zeros((num_traj, num_steps, 2)) # (5000, 800,2)
    angles=np.zeros((num_traj, num_steps, 1))

    env=RatSimulator(num_steps)

    print(">>Generating trajectory")
    for i in range(num_traj):
        vel, angVel, pos, angle =env.generateTrajectory()
        X[i,:,0]=vel
        X[i,:,1]=np.sin(angVel)
        X[i,:,2]=np.cos(angVel)
        positions[i,:]=pos

    init_X=np.zeros((num_traj,8,pcu + hcu))
    for i in range(8):
        init_X[:, i, :pcu]=dataGenerator.computePlaceCellsDistrib(positions[:,(i*100)], place_cell_centers)
        init_X[:, i, pcu:]=dataGenerator.computeHeadCellsDistrib(angles[:,(i*100)], head_cell_centers)


    print(">>Computing Actvity maps")
    #Feed 500 examples at time to avoid memory problems. Otherwise (10000*100=1million matrix)
    batch_size=500
    for startB in range(0, num_traj, batch_size): # num_traj=5000, batch_size=500
        endB=startB+batch_size

        #Divide the sequence in 100 steps.
        for startT in range(0, num_steps, 100):
            endT=startT+100

            #Retrieve the inputs for the timestep
            xBatch=X[startB:endB, startT:endT]
            test_placeCellGroundData = init_X[startB:endB, (startT//100), : pcu]
            test_headCellGroundData = init_X[startB:endB,  (startT//100), pcu :]            

            test_outputModel = agent.model([xBatch, test_placeCellGroundData, test_headCellGroundData])                        

            linearNeurons = test_outputModel[2]             
            linearNeurons = linearNeurons.numpy()           
            # print("linearneurons", linearNeurons.shape[0]) 
            
            #Convert 500,100,2 -> 50000,2
            posReshaped=np.reshape(positions[startB:endB,startT:endT],(-1,2)) # (50000, 2)            
            
            #save the value of the neurons in the linear layer at each timestep
            for t in range(linearNeurons.shape[0]): # 512
                #Compute which bins are for each position
                bin_x, bin_y=(posReshaped[t]//factor).astype(int)
                # print("bin_x, bin_y :", bin_x, bin_y)

                if(bin_y==bins):
                    bin_y=bins-1
                elif(bin_x==bins):
                    bin_x=bins-1

                #Now there are the 512 values of the same location
                activityMap[:,bin_y, bin_x] += np.abs(linearNeurons[t, 90,:]) #linearNeurons must be a vector of 512
                counterActivityMap[:, bin_y, bin_x] += np.ones((512))
    
    #Compute average value
    with np.errstate(divide = 'ignore', invalid='ignore'):
        result = np.true_divide(activityMap, counterActivityMap)
        result[result == np.inf] = 0
        result = np.nan_to_num(result)            
            
    os.makedirs("activityMaps", exist_ok=True)
    os.makedirs("corrMaps", exist_ok=True)

    #normalize total or single?
    normMap=(result -np.min(result))/(np.max(result)-np.min(result))
    #adding absolute value
    
    cols=16
    rows=32
    #Save images
    fig=plt.figure(figsize=(80, 80))
    for i in range(llu):
        fig.add_subplot(rows, cols, i+1)
        #normMap=(result[i]-np.min(result[i]))/(np.max(result[i])-np.min(result[i]))
        plt.imshow(normMap[i], cmap="jet", origin="lower")
        plt.axis('off')

    fig.savefig('activityMaps/neurons.jpg')

    fig=plt.figure(figsize=(80, 80))
    for i in range(llu):
        fig.add_subplot(rows, cols, i+1)
        #normMap=(result[i]-np.min(result[i]))/(np.max(result[i])-np.min(result[i]))
        plt.imshow(correlate2d(normMap[i], normMap[i]), cmap="jet", origin="lower")
        plt.axis('off')

    fig.savefig('corrMaps/neurons.jpg')






