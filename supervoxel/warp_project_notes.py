    Data: grid of voxels, each has its own starting label

    /*
    takes one voxel from curses import use_default_colors
from functools import total_ordering
from types import ClassMethodDescriptorType
from Data and evaluates its neighberhood
    class is differentiable, 
    */
    class diff_step()
        function forward(voxel,params,output)
            #looking in each direction
            maxValue=0
            for dir in ["up","down","left","right","ant","post"]
                #taking set of nearby voxels in given direction
                neigh=take_neighberhood(dir)
                #using params/ weights evaluate local model and saves maximum to maxValue
                maxValue=local_model(params, neigh)
            #saving output from all voxels
            output[x,y,z]=maxValue    
        
        function backwards()
            ...
            return gradient #gradient describing how parameters affect output
    
    /*
    takes output from last iteration and give the loss the smaller the better
    */
    function loss (output)
        ...
        loss


    #applying function to each voxel ald getting output
    output = data|> diff_step(params)
    
    lastLoss=0
    #loop until the change in loss function will not be smaller than epsilon
    #each time we are overwriting the output and using the same set of parameters to perform decision
    While(lastLoss-currLoss>e): 
        output = data|> diff_step(params)
        currLoss= loss(output)
    ... 
    #given parameters and final loss updates values of the parameters 
    new_params=optimize(params,final_loss)
    ...iterate


### Architecture
so diffStep need to be implemented as a module some properties
    outpus of this module will be ovewriten
        Hovewer gradients will be stored in the layer object 
    the weights in simpler design will not be shared, Although becouse of the depth some skip connections would be use_default_colors
    the gradients are always in respect to the input ad the inputs are the labels of each node 


loss(output)
    if(onEdge)#if it is next to the voxel of some other class
        atomic add loss total
    normalize by number of total voxels and some coefficient

    get mean variance in each class and later mean variance between classes     

    as a cost we return scaled and normalized number of classes plus mean of mean variances plus number of voxels on edge 