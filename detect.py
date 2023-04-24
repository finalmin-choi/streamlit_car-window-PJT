import matplotlib.pyplot as plt

colors = [(89, 161, 197),(67, 161, 255),(19, 222, 24)]

for color in colors:
    r, g, b = color
    plt.imshow([[(r/255, g/255, b/255)]])
    plt.show()


