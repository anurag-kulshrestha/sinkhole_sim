# Author: Anurag Kulshrestha
# Date created: 22-06-2019
# Last modified: 01-07-2019
# Purpose: To simulate sinkhole evolution and visualize them in pyqygraph

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import cv2


import matplotlib.pyplot as plt
import sys

#import pyqtgraph.parametertree.parameterTypes as pTypes
#from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType

#Initialization of the app
app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
w.setWindowTitle('Sinkhole simulation')
w.setCameraPosition(distance=200, elevation=-10, azimuth=40)

## Add a grid to the view: The grid would represent horizontal suface on the earth. The size of the each grid cell in this case is 20 meters x 20 meters

grid_cell_size_x, grid_cell_size_y = 5, 5 #meters
print('The grid cell size is = {} x {} meters'.format(grid_cell_size_x,grid_cell_size_y))

grid_dim_x, grid_dim_y = 50, 50 #number of cells
print('The total number of cells is = {} x {} meters'.format(grid_dim_x,grid_dim_y))

grid_total_extent_x, grid_total_extent_y = grid_cell_size_x*grid_dim_x, grid_cell_size_y*grid_dim_y
print('The total extent of the grid = {} x {} meters'.format(grid_total_extent_x,grid_total_extent_y))

g = gl.GLGridItem()
g.scale(grid_cell_size_x,grid_cell_size_y,1) # The scale of the grid is dependent on the grid_cell_size
g.setSize(grid_dim_x, grid_dim_x, 1) # The dimensions of the grid = 20 * 20.
g.setDepthValue(10)  # draw grid after surfaces since they may be translucent
w.addItem(g)

# To define the axes reference lines of the cartasian coordinate system
ax = gl.GLAxisItem()
ax.setSize(100,100,100)
w.addItem(ax)

def gauss_2D(xy,x_0,y_0,I,theta,sigma_x, sigma_y): 
    # function to find the compute the gaussian model
    # The parameters of the model are: 
    # 1) x_0, y_0: center coordinates of the 2D gaussian: 
    # 2) sigma_x, sigma_y: standard deviations of the model in 2 orthogonal horizontal directions
    # 3) theta: angle made by the major axis of the ellipse to the horizontal axis
    # 4) I: the rate of movement of the center of the gaussian model. Options: i) linear, ii) exponential
    
    x,y=xy
    x_dash=(x-x_0)*np.cos(theta) - (y-y_0)*np.sin(theta) #applying translational and rotational transformation
    y_dash=(x-x_0)*np.sin(theta) + (y-y_0)*np.cos(theta)
    if(isinstance(I, int)):
        z=I*np.exp(-0.5*((x_dash/sigma_x)**2 + (y_dash/sigma_y)**2))
    else:
        z= I.reshape((I.shape[0],1,1)) * np.exp(-0.5*((x_dash/sigma_x)**2 + (y_dash/sigma_y)**2))[np.newaxis,...]
    return z


#Modelling of the gaussian surface. The resolutution of the gaussian surface simulation is 1 meter.

extent = (100,100) # pixels
center = (50,50) # (pixel_num_x, pixel_num_y)
theta = np.pi/6 #radians
sigma_x = 5 # meters
sigma_y = 10 # meters

#Definition of the mesh
X_vec=np.arange(0,extent[0])
Y_vec=np.arange(0,extent[1])
XY=np.meshgrid(Y_vec,X_vec)

# Definition of the 
#I = 20 # amplitude of the gaussian   -> has to be replaced by linear movement/exponential movement.
temporal_resolution = 3#days
rate_of_deformation = -0.005 # meters per days
max_depth = -15 #meters
start_depth = 0

I_linear = np.arange(start_depth,max_depth,rate_of_deformation*temporal_resolution) # (1,000 iterations) linear rate of depression = 5 mm per day
I_exp = -np.logspace(0, 4, num=100, base=2)
Z = gauss_2D(XY, center[0], center[1], I_linear, theta, sigma_x,sigma_y)


## create a surface plot, tell it to use the 'heightColor' shader
## since this does not require normal vectors to render (thus we 
## can set computeNormals=False to save time when the mesh updates)

surf = gl.GLSurfacePlotItem(x=X_vec, y = Y_vec, shader='heightColor', computeNormals=False, smooth=False)
surf.shader()['colorMap'] = np.array([0.2, 3, 0.5, 0.2, 3, 1, 0.2, 0, 2])

surf.translate(10, 10, 0)
w.addItem(surf)

index = 0 # index for iteration of simulation video frames
#annotation
font = cv2.FONT_HERSHEY_SIMPLEX #font for annotation putText
line_sep = 20 #pixels. Distance between two lines in the annotation
def update():
    global surf, z, index
    index += 1
    surf.setData(z=Z[index%Z.shape[0]])
    img_name = 'test_{0:0=4d}.png'.format(index)
    w.grabFrameBuffer().save(img_name)
    
    #annotation of the image
    img = cv2.imread(img_name)
    
    cv2.putText(img,'Parameter Values:',(10,300), font, 0.7,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,'Center coordinates: ({},{})'.format(center[0], center[1]),(10,300+line_sep), font, 0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,'(Sigma_x, Sigma_y): ({},{})'.format(sigma_x, sigma_y),(10,300+2*line_sep), font, 0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,'Max depth of center: {} meters'.format(15),(10,300+3*line_sep), font, 0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,'Total simulation period: {} days'.format(int(max_depth/rate_of_deformation)),(10,300+4*line_sep), font, 0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,'Temporal Resolution: {} days'.format(temporal_resolution),(10,300+5*line_sep), font, 0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,'Velocity of center: {} millimeters/day'.format(5),(10,300+6*line_sep), font, 0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,'Day: {}'.format(index%Z.shape[0]*3),(10,300+7*line_sep), font, 0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.imwrite(img_name,img)
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(100)



## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
