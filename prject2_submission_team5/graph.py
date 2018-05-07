import matplotlib.pyplot as plt 

x = [1,3,5]
y1 = [76.55, 80.62, 95.48]
y2 = [65.37, 69.77, 97.73]

plt.plot(x, y1, marker='*', label='no autoencoder')
plt.plot(x, y2, marker='v', label='autoencoder')
plt.title('Comparision Graph')
plt.xlabel('Range')
plt.ylabel('Accuracy')
plt.legend()
plt.show()