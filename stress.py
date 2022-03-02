import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import argparse
import os

COLUMNS = ['x', 'y']
COLUMNS_LOG = ['log_x', 'log_y']
T_TABLE_COLUMNS = [50, 60, 70, 80, 90, 95, 98, 99, 99.5, 99.8, 99.9]
T_TABLE = {
    1:	[1.000,	1.376,	1.963,	3.078,	6.314,	12.71,	31.82,	63.66,	127.3,	318.3,	636.6],
    2:	[0.816,	1.080,	1.386,	1.886,	2.920,	4.303,	6.965,	9.925,	14.09,	22.33,	31.60],
    3:	[0.765,	0.978,	1.250,	1.638,	2.353,	3.182,	4.541,	5.841,	7.453,	10.21,	12.92],
    4:	[0.741,	0.941,	1.190,	1.533,	2.132,	2.776,	3.747,	4.604,	5.598,	7.173,	8.610],
    5:	[0.727,	0.920,	1.156,	1.476,	2.015,	2.571,	3.365,	4.032,	4.773,	5.893,	6.869],
    6:	[0.718,	0.906,	1.134,	1.440,	1.943,	2.447,	3.143,	3.707,	4.317,	5.208,	5.959],
    7:	[0.711,	0.896,	1.119,	1.415,	1.895,	2.365,	2.998,	3.499,	4.029,	4.785,	5.408],
    8:	[0.706,	0.889,	1.108,	1.397,	1.860,	2.306,	2.896,	3.355,	3.833,	4.501,	5.041],
    9:	[0.703,	0.883,	1.100,	1.383,	1.833,	2.262,	2.821,	3.250,	3.690,	4.297,	4.781],
    10:	[0.700,	0.879,	1.093,	1.372,	1.812,	2.228,	2.764,	3.169,	3.581,	4.144,	4.587],
    11:	[0.697,	0.876,	1.088,	1.363,	1.796,	2.201,	2.718,	3.106,	3.497,	4.025,	4.437],
    12:	[0.695,	0.873,	1.083,	1.356,	1.782,	2.179,	2.681,	3.055,	3.428,	3.930,	4.318],
    13:	[0.694,	0.870,	1.079,	1.350,	1.771,	2.160,	2.650,	3.012,	3.372,	3.852,	4.221],
    14:	[0.692,	0.868,	1.076,	1.345,	1.761,	2.145,	2.624,	2.977,	3.326,	3.787,	4.140],
    15:	[0.691,	0.866,	1.074,	1.341,	1.753,	2.131,	2.602,	2.947,	3.286,	3.733,	4.073],
    16:	[0.690,	0.865,	1.071,	1.337,	1.746,	2.120,	2.583,	2.921,	3.252,	3.686,	4.015],
    17:	[0.689,	0.863,	1.069,	1.333,	1.740,	2.110,	2.567,	2.898,	3.222,	3.646,	3.965],
    18:	[0.688,	0.862,	1.067,	1.330,	1.734,	2.101,	2.552,	2.878,	3.197,	3.610,	3.922],
    19:	[0.688,	0.861,	1.066,	1.328,	1.729,	2.093,	2.539,	2.861,	3.174,	3.579,	3.883],
    20:	[0.687,	0.860,	1.064,	1.325,	1.725,	2.086,	2.528,	2.845,	3.153,	3.552,	3.850],
    21:	[0.686,	0.859,	1.063,	1.323,	1.721,	2.080,	2.518,	2.831,	3.135,	3.527,	3.819],
    22:	[0.686,	0.858,	1.061,	1.321,	1.717,	2.074,	2.508,	2.819,	3.119,	3.505,	3.792],
    23:	[0.685,	0.858,	1.060,	1.319,	1.714,	2.069,	2.500,	2.807,	3.104,	3.485,	3.767],
    24:	[0.685,	0.857,	1.059,	1.318,	1.711,	2.064,	2.492,	2.797,	3.091,	3.467,	3.745],
    25:	[0.684,	0.856,	1.058,	1.316,	1.708,	2.060,	2.485,	2.787,	3.078,	3.450,	3.725],
    26:	[0.684,	0.856,	1.058,	1.315,	1.706,	2.056,	2.479,	2.779,	3.067,	3.435,	3.707],
    27:	[0.684,	0.855,	1.057,	1.314,	1.703,	2.052,	2.473,	2.771,	3.057,	3.421,	3.690],
    28:	[0.683,	0.855,	1.056,	1.313,	1.701,	2.048,	2.467,	2.763,	3.047,	3.408,	3.674],
    29:	[0.683,	0.854,	1.055,	1.311,	1.699,	2.045,	2.462,	2.756,	3.038,	3.396,	3.659],
    30:	[0.683,	0.854,	1.055,	1.310,	1.697,	2.042,	2.457,	2.750,	3.030,	3.385,	3.646],
    # 40:	[0.681,	0.851,	1.050,	1.303,	1.684,	2.021,	2.423,	2.704,	2.971,	3.307,	3.551],
    # 50:	[0.679,	0.849,	1.047,	1.299,	1.676,	2.009,	2.403,	2.678,	2.937,	3.261,	3.496],
    # 60:	[0.679,	0.848,	1.045,	1.296,	1.671,	2.000,	2.390,	2.660,	2.915,	3.232,	3.460],
    # 80:	[0.678,	0.846,	1.043,	1.292,	1.664,	1.990,	2.374,	2.639,	2.887,	3.195,	3.416],
    # 100:	[0.677,	0.845,	1.042,	1.290,	1.660,	1.984,	2.364,	2.626,	2.871,	3.174,	3.390],
    # 120:	[0.677,	0.845,	1.041,	1.289,	1.658,	1.980,	2.358,	2.617,	2.860,	3.160,	3.373],
    1000:	[0.674,	0.842,	1.036,	1.282,	1.645,	1.960,	2.326,	2.576,	2.807,	3.090,	3.291],
}

class LinearRegression:
    def __init__(self):
        pass

    def fit(self, X, y):
        print('[*] fitting with linear regression...')
        self.w = np.matmul(np.matmul(np.linalg.pinv(np.matmul(X.T, X)), X.T), y)  # (X.T X)^-1 X^T y

        pred = self.predict(X)

        self.sigma = np.sum(np.square(y - pred)) / (len(X) - 2)

    def predict(self, X):
        if self.w is None:
            raise Exception('Linear regression model has not been trained!')

        pred = np.matmul(X, self.w)[:, 0]

        return pred

    def predict_with_conf(self, X, conf):
        t_table = pd.DataFrame.from_dict(T_TABLE, orient='index', columns=T_TABLE_COLUMNS).T
        try:
            t = t_table.loc[conf, len(X)-1]
        except:
            t = t_table.loc[conf, 1000]

        pred = self.predict(X)
        X = X[:, 0]
        X_mean = np.mean(X)

        conf = t * np.sqrt(self.sigma * (1. / len(X) +
                      (np.power(X - X_mean, 2) / np.sum(np.power(X - X_mean, 2)))))
        lower = pred - conf
        upper = pred + conf

        return pred, lower, upper

def sanitize_parameters(fpath, conf):
    if fpath.endswith('/') or fpath.endswith('\\'):
        fpath = fpath[:-1]

    if not os.path.exists(fpath):
        print('[!] Invalid path. please check and write the exact path to data file.')
        exit(1)

    if conf not in [90, 95, 99]:
        print('[!] only allowed 90, 95, 99 for confidence interval.')
        exit(1)

def load_data(path):
    print('[*] loading data...')

    if '.csv' in pathlib.Path(path).suffix.lower():
        data = pd.read_csv(path, header=None)
    else:
        print('invalid file format for data. please make sure using CSV.')
        exit(1)

    if data.shape[1] > 2:
        data = data.iloc[:, :2]

    data.columns = COLUMNS

    return data

def log_scale(data):
    print('[*] scaling with log10...')

    data = np.log10(data)
    data.columns = COLUMNS_LOG

    return data

def plot(data, model, conf, fn):
    print('[*] plotting...')

    min_x, max_x = data['log_x'].min(), data['log_x'].max()
    x_space = np.linspace(min_x, max_x, 100)[:, np.newaxis]
    x_space = np.concatenate((x_space, np.ones_like(x_space)), 1)

    y_space, lower, upper = model.predict_with_conf(x_space, conf)
    x_space = x_space[:, 0]

    fig = plt.figure()
    plt.scatter(data['log_y'], data['log_x'], edgecolors='k', facecolor='none', label='data')
    plt.plot(y_space, x_space, c='b', label='prediction')
    plt.plot(lower, x_space, color='r', linewidth=0.5, linestyle='dashed', label='%.2f %% confidence interval'%(conf))
    plt.plot(upper, x_space, color='r', linewidth=0.5, linestyle='dashed')
    plt.ylabel('Stress (MPa)')
    plt.xlabel('Time (h)')
    plt.legend()
    plt.savefig(f'{fn}-plot.png')
    fig.show()

def calc_F(model, data):
    mean_y_by_X = data.groupby('log_x').mean().reset_index()
    mean_y_by_X.columns = ['log_x', 'mean_log_y']

    X = data['log_x'].to_numpy()[:, np.newaxis]
    X = np.concatenate((X, np.ones_like(X)), 1)
    y = data['log_y'].to_numpy()
    pred = model.predict(X)

    SSh = np.sum(np.square(y - pred))
    x_y_mean_y = pd.merge(data, mean_y_by_X, on='log_x', how='inner')
    SSe = np.sum(np.square((x_y_mean_y.iloc[:, 1] - x_y_mean_y.iloc[:, 2]).to_numpy()))

    Vh = float(data.shape[0] - 2)
    Ve = float(data.shape[0] - mean_y_by_X.shape[0])

    F = ((SSh - SSe) / (Vh - Ve)) / (SSe / Ve)
    return F

def main(fpath, conf, is_visual):
    data = load_data(fpath)
    data = log_scale(data)
    model = LinearRegression()
    # the purpose of this model is to predict the time for corresponding to stress(MPa).
    X = data.iloc[:, 0].to_numpy()[:, np.newaxis]
    X = np.concatenate((X, np.ones_like(X)), 1)
    y = data.iloc[:, 1].to_numpy()[:, np.newaxis]
    model.fit(X, y)
    w = model.w
    F = calc_F(model, data)

    print()
    print('[*] gradient:', w[0, 0])
    print('[*] bias:', w[1, 0])
    print('[*] statistics (F):', F)
    print()

    pd_data = {
        'gradient': [w[0, 0]],
        'bias': [w[1, 0]],
        'statistics (F)': [F]
    }
    fn = pathlib.Path(fpath).stem
    pd_data = pd.DataFrame(data=pd_data).T
    pd_data.to_excel(f'{fn}-output.xlsx', header=False)

    plot(data, model, conf, fn)

    if is_visual:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script learn the stress test model and visualize it.')
    parser.add_argument('--fpath', type=str, required=True, help='directory including data files.')
    parser.add_argument('--conf', type=float, required=True, help='the confidence level of t-distribution: 90, 95, 99')
    parser.add_argument('-V', help='visual mode for the results.', action='store_true')

    args = parser.parse_args()
    fpath, conf, is_visual = args.fpath, args.conf, args.V

    sanitize_parameters(fpath, conf)

    print('[*] path to file:', fpath)
    print('[*] level of confidence interval:', conf)
    print('[*] in visual mode:', is_visual)
    print()

    main(fpath, conf, is_visual)
