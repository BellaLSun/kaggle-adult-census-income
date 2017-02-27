# encoding: UTF-8

from data.cn_to_ctn import cn_to_ctn
import numpy as np
import pandas as pd
from pandas_summary import DataFrameSummary


def read_income_data(dummies=False):
    # Load the file
    df = pd.read_csv("data/adult-census-income.zip", header=0)

    df = convert_target_to_float(df)
    df = convert_sex_to_float(df)
    df = add_continent_info(df)
    df = add_is_single_info(df)
    df = merge_capital_columns(df)
    df = delete_useless_columns(df)
    df = set_null_values(df)

    if dummies:
        df = pd.get_dummies(df)

    return df


def convert_target_to_float(df):
    df.loc[df.income == "<=50K", "income"] = 0.0
    df.loc[df.income == ">50K", "income"] = 1.0
    df.income = df.income.astype(float)
    return df


def convert_sex_to_float(df):
    df.loc[df.sex == "Male", "sex"] = 0.0
    df.loc[df.sex == "Female", "sex"] = 1.0
    df.sex = df.sex.astype(float)
    return df


def add_continent_info(df):
    df['native.continent'] = df['native.country'].apply(lambda x: cn_to_ctn.get(x))
    return df


def add_is_single_info(df):
    df['is_single'] = 1
    av_spouse_bool = df['marital.status'] == 'Married-AF-spouse'
    civ_spouse_bool = df['marital.status'] == 'Married-civ-spouse'
    df.loc[av_spouse_bool | civ_spouse_bool, 'is_single'] = 0
    return df


def merge_capital_columns(df):
    df['capital'] = df['capital.gain'] - df['capital.loss']
    df['capital.log'] = np.log2(df['capital.gain'] + 1) - np.log2(df['capital.loss'] + 1)
    df['capital.log.round'] = np.round(df['capital.log'])
    return df


def delete_useless_columns(df):
    df.pop("education")
    df.pop('capital.gain')
    df.pop('capital.loss')
    return df


def set_null_values(df):
    df = df[-df.isin(["?"])]
    return df


def get_summary(df):
    return DataFrameSummary(df)
