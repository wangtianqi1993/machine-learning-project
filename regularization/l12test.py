# !usr/bin/env python
# -coding -*-utf-8-*-
__author__ = 'wtq'

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston


def l1_regular():
    boston = load_boston()
    scaler = StandardScaler()
    x = scaler.fit_transform(boston['data'])
    y = boston['target']
    names = boston['feature_names']

    lasso = Lasso(alpha=.3)
    lasso.fit(x, y)
    # print 'lasso model: ', pretty_print_linear(lasso.coef_, names, sort = True)


if __name__ == "__main__":
    l1_regular()

