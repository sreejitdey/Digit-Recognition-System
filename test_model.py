import imageio
import numpy as np
from matplotlib import pyplot as plt

img = imageio.imread("image0.png")
gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.show()

gray = gray.reshape(-1, 28, 28, 1)
gray = gray / 255

from keras.models import load_model
model = load_model("model.h5")

prediction = model.predict(gray)
print(prediction.argmax())
