import numpy as np

def get_start_indices(sequences):
    indices = [0]
    for i in range(1,len(sequences)):
        if (sequences[i]==4) and (sequences[i-1]==6):
            indices.append(i)
    indices.append(len(sequences))
    return indices

def group_seizure(X, y, sequences):
    Xg = []
    yg = []
    start_indices = get_start_indices(sequences)
    print ('start_indices', start_indices)
    print (len(X), len(y))
    for i in range(len(start_indices)-1):
        Xg.append(
            np.concatenate(X[start_indices[i]:start_indices[i+1]], axis=0)
        )
        yg.append(
            np.array(y[start_indices[i]:start_indices[i+1]])
        )
    return Xg, yg
