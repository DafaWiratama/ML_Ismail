import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

if __name__ == '__main__':
    df = pd.read_csv("jalan.csv")
    print()

    plt.title("Rotation Vector")
    plt.plot(df.rotation_vector_x, '--', label="x")
    plt.plot(df.rotation_vector_y, '--', label="y")
    plt.plot(df.rotation_vector_z, '--', label="z")
    plt.plot(df.rotation_vector_scalar, '--', label='scalar')
    plt.legend()
    plt.show()

    plt.title("Acceleration")
    plt.plot(df.acceleration_x, '--', label='x')
    plt.plot(df.acceleration_y, '--', label='y')
    plt.plot(df.acceleration_z, '--', label='z')
    plt.legend()
    plt.show()

    plt.title("Gravity")
    plt.plot(df.gravity_x, '--', label='x')
    plt.plot(df.gravity_y, '--', label='y')
    plt.plot(df.gravity_z, '--', label='z')
    plt.legend()
    plt.show()

    plt.title("Magnetic")
    plt.plot(df.magenetic_x, '--', label='x')
    plt.plot(df.magenetic_y, '--', label='y')
    plt.plot(df.magenetic_z, '--', label='z')
    plt.legend()
    plt.show()
