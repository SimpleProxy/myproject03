#!/usr/bin/python3

from time import sleep
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]]
    # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 100,
               facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        # Warning: passing non-integers as arguments is deprecated
        plt.subplot(int(nGraphRow), nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    # saving plot to output directory
    outputFig = outputDir + "plotPerColumnDistribution.png"
    plt.savefig(outputFig, dpi=300, format="png")
    plt.clf()

def plotCorrelationMatrix(df, graphWidth):
    #filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    # keep columns where there are more than 1 unique values
    df = df[[col for col in df if df[col].nunique() > 1]]
    if df.shape[1] < 2:
        print(f"No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2")
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth),
                dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {inputDir}', fontsize=25)

    # auto scale axis
    plt.autoscale(enable=True, axis="x", tight=True)
    plt.autoscale(enable=True, axis="y", tight=True)


    # saving plot to output directory
    outputFig = outputDir + "plotCorrelationMatrix.png"
    plt.savefig(outputFig, dpi=300, format="png")
    plt.clf()

# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    # keep columns where there are more than 1 unique values
    df = df[[col for col in df if df[col].nunique() > 1]]
    columnNames = list(df)

    # reduce the number of columns for matrix inversion of kernel density plots
    if len(columnNames) > 10:
        columnNames = columnNames[:10]
    df = df[columnNames]

    plt.xticks(rotation=90)
    plt.yticks(rotation=90)

    ax = pd.plotting.scatter_matrix(df, alpha=0.75,
                                    figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2),
                            xycoords='axes fraction', ha='center', va='center',
                            size=textSize)
    plt.suptitle('Scatter and Density Plot')

    # saving plot to output directory
    outputFig = outputDir + "plotScatterMatrix.png"
    plt.savefig(outputFig, dpi=300, format="png")
    plt.clf()

def ecdf(data):
    n = len(data)

    x = np.sort(data)
    y = np.arange(1, n + 1) / n

    return x, y

def plot_ecdf(data):
    xAxis, yAxis = ecdf(data)

    plt.xlabel("Nationalitty")
    plt.ylabel("ECDF")

    plt.xticks(rotation=90)

    # auto scale axis
    plt.autoscale(enable=True, axis="x", tight=True)
    plt.autoscale(enable=True, axis="y", tight=True)


    plt.plot(xAxis, yAxis, marker = ".", linestyle = "none")

    # saving plot to output directory
    outputFig = outputDir + "ecdf.png"
    plt.savefig(outputFig, dpi=300, format="png")
    plt.clf()

# gender: student's gender
# NationalITy: student's nationallity
# PlaceofBirth: place where student was born
# StageID: educational level student belongs to (lowerlevel, middleschool, highschool)
# GradeID: grade student belongs to from G-01 to G-12
# SectionID: classroom student belongs
# Topic: course topic
# Semester: school year semester
# Relation: which parent is responsible for the student
# raisedhands: times students interact during class
# VisITedResources: how many times a stundet visit content
# AnnouncementsView: how many times a student check for announcements
# Discussion: how many times a stundet participate in group discussion
# ParentAnsweringSurvey:
# ParentschoolSatisfaction:
# StudentAbsenceDays:
# Class: the measured perfomance in exams

if __name__ == "__main__":

    inputDir = "./input.d/xAPI-Edu-Data.csv"
    outputDir = "./output.d/"

    data = pd.read_csv(inputDir)
    df1 = pd.DataFrame(data)

    df2 = df1
    df2["gender"] = df2["gender"].replace(["F", "M"], [0, 1])
    df2["Semester"] = df2["Semester"].replace(["First", "Second"], [0, 1])
    df2["Relation"] = df2["Relation"].replace(["mom", "father"], [0, 1])
    df2["ParentschoolSatisfaction"] = df2["ParentschoolSatisfaction"].replace(["Bad",
        "Good"], [0, 1])
    df2["ParentAnsweringSurvey"] = df2["ParentAnsweringSurvey"].replace(["No", "Yes"], [0, 1])
    df2["StudentAbsenceDays"] = df2["StudentAbsenceDays"].replace(["Above-7", "Under-7"], [0, 1])


    plotPerColumnDistribution(df1, 10, 5)
    sleep(2)
    plotScatterMatrix(df1, 12, 10)
    sleep(2)
    plotCorrelationMatrix(df2, 12)

