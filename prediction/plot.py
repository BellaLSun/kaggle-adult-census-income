# encoding: UTF-8

import matplotlib
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

matplotlib.style.use('ggplot')


def bar_plot(df, attr):
    df.groupby([attr, 'income']).size().unstack().plot.bar(stacked=True)


def bar_percentage_plot(df, attr):
    sums = df.groupby([attr, 'income']).size()
    percents = sums.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
    percents.unstack().plot.bar(stacked=True)


def scatter_plot(df, attr_x, attr_y):
    df.plot.scatter(x=attr_x, y=attr_y, c='income', s=10)


def hexbin_plot(df, attr_x, attr_y):
    df.plot.hexbin(x=attr_x, y=attr_y, C='income')


def scatter_matrix_plot(df, attrs):
    attrs = ['age', 'fnlwgt', 'hours.per.week', 'education.num']
    scatter_matrix(
        df[attrs],
        alpha=0.2,
        figsize=(6, 6),
        diagonal='kde'
    )
