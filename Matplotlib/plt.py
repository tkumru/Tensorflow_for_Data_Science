import numpy as np
import matplotlib.pyplot as plt

nparray1 = np.linspace(0, 10, 20)
nparray2 = nparray1 ** 2

plt.subplot(1, 2, 1)
plt.plot(nparray1, nparray2, 'g--')
plt.subplot(1, 2, 2)
plt.plot(nparray1, nparray2, 'r*-')
plt.show()

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.9, 0.9])
ax.plot(nparray1, nparray1, label="numpy array")
ax.plot(nparray1, nparray1 ** 2, label="numpy array ** 2")
ax.plot(nparray1, nparray1 ** 3, label="numpy array ** 3")
ax.legend()
plt.show()

plt.scatter(nparray1, nparray2)
plt.show()

nparray3 = np.random.randint(0, 100, 50)
plt.hist(nparray3)
plt.show()

plt.boxplot(nparray3)
plt.show()
