import spiralcreator 
import numpy as np
from matplotlib import pyplot as plt
points, classes = spiralcreator.generate_two_spirals_dataset(5, 40)

points = np.array(points)
Xx = points.T[0]
Xy = points.T[1]
classes = np.array(classes)
Y = classes.T
C = list(map(lambda c: 0.5 * c, Y))
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.scatter(Xx, Xy, c=C, s=50, alpha=0.75)
ax1.set_xlabel('Y labels')
plt.show()