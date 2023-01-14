import numpy as np
import pandas as pd


def get_train_x_train_y(train_df: pd.DataFrame):
    fc = 18  # feature count

    # get year data
    year_data = list()
    for month in range(12):  # 0 - 11
        total_hr = 24 * 20
        temp = np.zeros((fc, total_hr))

        day_per_month = 20
        for day in range(day_per_month):
            hr_idx = 24 * day
            row_idx = 18 * 20 * month + 18 * day
            temp[:, hr_idx: hr_idx + 24] = train_df.iloc[row_idx: row_idx + 18]

        year_data.append(temp)

    year_data = np.array(year_data)

    train_x, train_y = list(), list()

    for month in range(12):
        month_data = year_data[month]
        for hr_itv_idx in range(24 * 20 - 9):
            x = month_data[:, hr_itv_idx: hr_itv_idx + 9].flatten()
            y = month_data[9, hr_itv_idx + 9]  # pm2.5 is at row-9

            train_x.append(x)
            train_y.append(y)

    return np.array(train_x), np.array(train_y)
