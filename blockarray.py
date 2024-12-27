import numpy as np

class BlockArray(np.ndarray):
    def __new__(cls, input_array, xF_dim, xB_dim=0,eta_dim=0):
        obj = np.asarray(input_array).view(cls)
        obj.__init__(input_array, xF_dim)
        return obj

    def __init__(self, input_array, xF_dim, xB_dim=0,eta_dim=0):
        self.xF_dim =xF_dim
        self.xB_dim=xB_dim
        self.eta_dim=eta_dim
    def __getitem__(self, index):
        if isinstance(index, (list, np.ndarray)):
            # If a list or numpy array of integers is provided, return a list of vector chunks of size 3
            assert len(index) <= 2, "Only a single and dual index is supported"
            if len(index)==1:
                row=index[0]
                if row==-1:
                    # we are indexing the robot pose
                    row_start=0; row_stop = row_start + self.eta_dim
                elif row==-2:
                    # we are indexing the robot state
                    row_start=0; row_stop = row_start + self.xB_dim
                else:
                    row_start = self.xB_dim + row * self.xF_dim;
                    row_stop = row_start + self.xF_dim
                # start = self.xB_dim + index[0] * self.xF_dim
                # stop = start + self.xF_dim
                v = super(BlockArray, self).__getitem__(slice(row_start, row_stop))
                assert v.size>0, "Bad Indexing"
                return v
            else:
                row,col=index

                if row==-1:
                    # we are indexing the robot pose
                    row_start=0; row_stop = row_start + self.eta_dim
                elif row==-2:
                    # we are indexing the robot state
                    row_start=0; row_stop = row_start + self.xB_dim
                else:
                    row_start = self.xB_dim + row * self.xF_dim;
                    row_stop = row_start + self.xF_dim

                if col==-1:
                    # we are indexing the robot pose
                    column_start=0; column_stop = column_start + self.eta_dim
                elif col==-2:
                    # we are indexing the robot state
                    column_start=0; column_stop = column_start + self.xB_dim
                else:
                    column_start = self.xB_dim + col * self.xF_dim;
                    column_stop = column_start + self.xF_dim

                v= super(BlockArray, self).__getitem__((slice(row_start, row_stop),slice(column_start,column_stop)))
                #assert v.shape==(self.xF_dim, self.xF_dim), "Bad Indexing"
                return v
        else:
            # For other index types, use the default behavior
            return super(BlockArray, self).__getitem__(index)


    def __setitem__(self, index, value):
        if isinstance(index, (list, np.ndarray)):
            # If a list or numpy array of integers is provided, return a list of vector chunks of size 3
            assert len(index) <= 2, "Only a single and dual index is supported"
            if len(index)==1:
                row=index[0]
                if row==-1:
                    # we are indexing the robot pose
                    row_start=0; row_stop = row_start + self.eta_dim
                elif row==-2:
                    # we are indexing the robot state
                    row_start=0; row_stop = row_start + self.xB_dim
                else:
                    row_start = self.xB_dim + row * self.xF_dim;
                    row_stop = row_start + self.xF_dim
                # start = self.xB_dim + index[0] * self.xF_dim
                # stop = start + self.xF_dim
                #super(BlockArray, self).__setitem__(slice(start, stop),value[0])
                self[row_start:row_stop]=value
            else:
                # row,col=index
                # row_start = self.xB_dim + index[0] * self.xF_dim; row_stop = row_start + self.xF_dim
                # column_start = self.xB_dim + index[1] * self.xF_dim; column_stop = column_start + self.xF_dim
                row,col=index

                if row==-1:
                    # we are indexing the robot pose
                    row_start=0; row_stop = row_start + self.eta_dim
                elif row==-2:
                    # we are indexing the robot state
                    row_start=0; row_stop = row_start + self.xB_dim
                else:
                    row_start = self.xB_dim + row * self.xF_dim;
                    row_stop = row_start + self.xF_dim

                if col==-1:
                    # we are indexing the robot pose
                    column_start=0; column_stop = column_start + self.eta_dim
                elif col==-2:
                    # we are indexing the robot state
                    column_start=0; column_stop = column_start + self.xB_dim
                else:
                    column_start = self.xB_dim + col * self.xF_dim;
                    column_stop = column_start + self.xF_dim
                #super(BlockArray, self).__setitem__((slice(row_start, row_stop),slice(column_start,column_stop)),value)
                self[row_start:row_stop,column_start:column_stop]=value
        else:
            # For other index types, use the default behavior
            super(BlockArray, self).__setitem__(index, value)

if __name__ == '__main__':

    # Create a NumPy array
    arr = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])

    # Create an instance of the custom array class
    my_array = BlockArray(arr.T,2)

    # Access and modify vector chunks of size 3 using my_array[1]
    print(my_array[1])  # Output: [4, 5, 6]

    print(my_array[[1]])  # Output: [[4, 5, 6], [7, 8, 9]]


    v=BlockArray(np.array([[1,2,3,4,5,6],[21,22,23,24,25,26]]),2)
    print(v[[0,1]])

    v[[0,1]]=np.array([[31,32],[41,42]])
    print(v[[0,1]])

