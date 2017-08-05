"""
GoodTecher Machine Learning Coding Tutorial
http://www.goodtecher.com

Machine Learning Coding Tutorial 3. What Makes a Good Feature?

The program takes a feature (dogs height) as input
and display normal distribution of types of dogs
"""

import numpy as np
import matplotlib.pyplot as plt

# creates 500 Greyhounds and 500 Labradors
number_of_greyhounds = 500
number_of_labradors = 500

# Greyhounds on average 28 inches tall
# Labradors on average 24 inches tall
# Let's say height is normally distributed,
# so we'll make both of these plus or minus 4 inches
# the following code generates arrays of 500 Greyhound heights
# and 500 Labradors heights
greyhounds_heights = 28 + 4 * np.random.randn(number_of_greyhounds)
labradors_heights = 24 + 4 * np.random.randn(number_of_labradors)

# visualize in histogram,
# Greyhounds are in red, Labradors are in blue
plt.hist([greyhounds_heights, labradors_heights], stacked=True, color=['r', 'b'])
plt.show()
