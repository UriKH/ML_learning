from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from cat_recognizer import predict


def plot_learning_curve(model):
    costs = np.squeeze(model['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title(f"Learning rate = {str(model['learning_rate'])}")
    plt.show()


def test_image(file_name, model, classes, num_px):
    image = np.array(Image.open(file_name).resize((num_px, num_px)))
    plt.imshow(image)
    image = image / 255.
    image = image.reshape((1, num_px * num_px * 3)).T
    my_predicted_image = predict(model["w"], model["b"], image)

    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" +
          classes[int(np.squeeze(my_predicted_image)), ].decode("utf-8") + "\" picture.")
