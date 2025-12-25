import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline


class PolynomialRegression:
    def __init__(self, degree=2):
        self.model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('scaler', StandardScaler()),
            ('linear', Ridge(fit_intercept=True))
        ])
        
    def fit(self, X, y):
        X = np.asarray(X).reshape(-1, 1) if X.ndim == 1 else X
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        X = np.asarray(X).reshape(-1, 1) if X.ndim == 1 else X
        return self.model.predict(X)

flag = 0.0
hermite_polynoms = [
    lambda x: np.exp(-0.5 * flag * x ** 2) * (x),
    lambda x: np.exp(-0.5 * flag * x ** 2) * (x ** 2 - 1),
    lambda x: np.exp(-0.5 * flag * x ** 2) * (x ** 3 - 3 * x),
    lambda x: np.exp(-0.5 * flag * x ** 2) * (x ** 4 - 6 * x ** 2 + 3),
    lambda x: np.exp(-0.5 * flag * x ** 2) * (x ** 5 - 10 * x ** 3 + 15 * x),
    lambda x: np.exp(-0.5 * flag * x ** 2) * (x ** 6 - 15 * x ** 4 + 45 * x ** 2 - 15),
    lambda x: np.exp(-0.5 * flag * x ** 2) * (x ** 7 - 21 * x ** 5 + 105 * x ** 3 - 105 * x)
]
    
class HermiteRegression:
    def __init__(self, degree=2, alpha=1):
        self.alpha = alpha
        self.degree = min(degree, len(hermite_polynoms))
        self.scaler = StandardScaler()
        self.model = Ridge(fit_intercept=True)
        
    def transform(self, X):
        X = np.stack([hermite_polynoms[i](self.alpha * X) for i in range(self.degree)], axis=-1)
        return X
    
    def fit(self, X, y):
        X = self.scaler.fit_transform(X.reshape(-1, 1)).reshape(-1)
        X = self.transform(X)
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        X = self.scaler.transform(X.reshape(-1, 1)).reshape(-1)
        X = self.transform(X)
        return self.model.predict(X)


EPS = 0.0
def fit_american_montecarlo(payoff, features, model_generator, POI=True, fit_itm=True, df=1):
    nt, nsim = payoff.shape
    # current price
    price = payoff[-1].copy()
    models = [model_generator() for _ in range(nt)]
    expiration_date = np.zeros([nsim], int) + (nt - 1)
    for t in range(nt - 2, -1, -1):
        # fit model
        in_the_money = np.ones(nsim, bool)
        price = df * price
        if fit_itm:
            in_the_money = payoff[t] > EPS

        if in_the_money.sum():
            models[t].fit(features[t, in_the_money], price[in_the_money])
            # predict: continuation value
        else:
            models[t].fit(features[t], EPS + 1 + np.ones(nsim))
        continuation = models[t].predict(features[t]) 
        # expiry now if in the money and payoff > continuation value
        expiry_now = in_the_money & (payoff[t] > continuation)
        price[expiry_now] = payoff[t, expiry_now]
        if not POI:
            price[~expiry_now] = continuation[~expiry_now]
        expiration_date[expiry_now] = t
    upper_bound = np.mean(np.max(payoff, axis=0))
    lower_bound = np.max(np.mean(payoff, axis=1))
    res = {'price': np.mean(price), 'model': models, 'std': np.std(price), 'prices': price,
           'upper_bound': upper_bound, 'lower_bound': lower_bound, 'expiration_date': expiration_date}
    return res


def predict_american_montecarlo(payoff, features, models):
    nt, nsim = payoff.shape
    price = []
    # mask for expired options
    expired = np.zeros([nsim], bool)
    expiration_date = np.zeros([nsim], int)
    for t in range(0, nt - 1):
        cont = models[t].predict(features[t]).clip(EPS)
        expiry_now = (~expired) & (payoff[t] > cont)
        price.extend(payoff[t, expiry_now].copy())
        expiration_date[expiry_now] = t
        expired |= expiry_now
    price.extend(payoff[-1, ~expired].clip(EPS))
    expiration_date[~expired] = nt - 1

    upper_bound = np.mean(np.max(payoff, axis=0))
    lower_bound = np.max(np.mean(payoff, axis=1))

    res = {'price': np.mean(price), 'std': np.std(price), 'euro': np.mean(payoff, axis=1),
           'upper_bound': upper_bound, 'lower_bound': lower_bound, 'expiration_date': expiration_date}
    return res