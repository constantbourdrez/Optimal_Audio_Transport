import numpy as np


def a_weighting_amp(freq):
    # Convert to hertz
    freq /= 2 * np.pi
    freq_squared = freq ** 2
    top = 12194 ** 2 * freq_squared ** 2
    bot1 = 20.6 ** 2 + freq_squared
    bot2 = 107.7 ** 2 + freq_squared
    bot3 = 737.9 ** 2 + freq_squared
    bot4 = 12194 ** 2 + freq_squared
    return top / (bot1 * np.sqrt(bot2 * bot3) * bot4)

def apply(points):
    for w in range(len(points)):
        for i in range(len(points[w])):
            points[w][i].value *= a_weighting_amp(points[w][i].freq)

def remove(points):
    for w in range(len(points)):
        for i in range(len(points[w])):
            value = a_weighting_amp(points[w][i].freq)
            if value > 0:
                points[w][i].value /= value
