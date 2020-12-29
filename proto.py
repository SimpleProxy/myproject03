#!/usr/bin/python3

import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt

inputDir = "./input.d/xAPI-Edu-Data.csv"
outputDir = "./output.d/"

data = pd.read_csv(inputDir)
#myArray = np.array(data)

df = pd.DataFrame(data)
#print(df, df.shape, df.describe(), df.info())

#TODO:  use values given by the initial exploration to create the label and axis
# of the plot

for i in df.columns:
    print(df[i].describe())

columns = ["StudentAbsenceDays", "Semester"]
for x in columns:
    n_data = len(df[x])
    n_bins = int(np.sqrt(n_data))

    plt.hist(df[x])
    plt.xlabel("X")
    plt.ylabel("Y")
#    plt.legend(loc="upper left")

    # auto scale axis
    plt.autoscale(enable=True, axis="x", tight=True)
    plt.autoscale(enable=True, axis="y", tight=True)

    #draw a grid
    plt.grid()

    outputFig = outputDir + x
    plt.savefig(outputFig, dpi=300, format="png")

    plt.clf() # clear current figure
#    plt.show()

