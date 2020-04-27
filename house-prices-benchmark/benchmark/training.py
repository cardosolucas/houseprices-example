from sklearn.linear_model import LassoCV

def training(data):
    lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005])
    model = lasso.fit(data['train_x'], data['train_y'])
    return model