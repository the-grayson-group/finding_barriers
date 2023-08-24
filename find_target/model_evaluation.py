import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, ExpSineSquared, Matern
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from reaction_repr import get_one_hot_encoding

def fit_model(model, train_set, parameter_ranges):
	grid_search = GridSearchCV(model, parameter_ranges, scoring="neg_mean_absolute_error", cv=5, refit=True, n_jobs=2)
	grid_search.fit(train_set[:,:-1], train_set[:,-1])
	return grid_search.best_estimator_

def evaluate_model(model, train_set, test_set, parameter_ranges):
	trained_model = fit_model(model, train_set, parameter_ranges)
	train_predictions = trained_model.predict(train_set[:,:-1])
	test_predictions = trained_model.predict(test_set[:,:-1])
	train_error = mean_absolute_error(train_set[:,-1], train_predictions)
	test_error = mean_absolute_error(test_set[:,-1], test_predictions)
	return train_error, test_error

def generate_datasets(Xy, train_size=30, rstate=1):
	Xy_new = Xy.copy()
	np.random.seed(rstate)
	np.random.shuffle(Xy_new)
	train_set = Xy_new[:train_size]
	test_set = Xy_new[train_size:]
	return train_set, test_set

def test_models(models, Xy, parameter_ranges, n_repeats=5):
	train_scores = np.array([0 for i in range(len(models))], dtype=float)
	test_scores = np.array([0 for i in range(len(models))], dtype=float)
	train_square_errors = np.array([0 for i in range(len(models))], dtype=float)
	test_square_errors = np.array([0 for i in range(len(models))], dtype=float)
	for r in range(1, n_repeats+1):
		train_set, test_set = generate_datasets(Xy, rstate=r)
		for i, (model, parameter_range) in enumerate(zip(models, parameter_ranges)):
			train_error, test_error = evaluate_model(model, train_set, test_set, parameter_range)
			train_scores[i] += train_error
			test_scores[i] += test_error
			train_square_errors[i] += train_error**2
			test_square_errors[i] += test_error**2
	return train_scores / n_repeats, test_scores / n_repeats, np.sqrt(np.abs((train_square_errors / n_repeats) - (train_scores / n_repeats)**2)), np.sqrt(np.abs((test_square_errors / n_repeats) - (test_scores / n_repeats)**2))


if __name__ == "__main__":
	models = [LinearRegression(), Ridge(), RandomForestRegressor(), GradientBoostingRegressor(), KernelRidge(), GaussianProcessRegressor()]
	parameter_ranges = [{"fit_intercept": [True, False]},
						{"alpha": [10**i for i in range(-7,3)]},
						{"n_estimators": [50, 100, 200], "max_depth": [i for i in range(3, 36, 8)]},
						{"n_estimators": [50, 100, 200], "max_depth": [i for i in range(3, 36, 8)], "learning_rate": [0.9, 0.1, 0.01, 0.001]},
						{"kernel": ["linear", "poly", "rbf", "sigmoid"], "alpha": [10**i for i in range(-7, 3)], "gamma": [10**i for i in range(-7, 3)], "degree": [2, 3, 4]},
						{"kernel": [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
						1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1, alpha_bounds=(1e-5, 1e15)),
						1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0, length_scale_bounds=(0.1, 10.0), periodicity_bounds=(1.0, 10.0)),
						1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5),
						1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)) + 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5),
						1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)) + 1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0, length_scale_bounds=(0.1, 10.0), periodicity_bounds=(1.0, 10.0))]}]
	datasets = ["barrier_feature_groups.csv", "vaska_barrier_feature_groups.csv", "e2_barrier_feature_groups.csv", "sn2_barrier_feature_groups.csv"]
	model_names = ["LR", "RR", "RFR", "GBR", "KRR", "GPR"]
	for dataset in datasets:
		print(dataset)
		Xy = get_one_hot_encoding(dataset)
		train_scores, test_scores, train_std_devs, test_std_devs = test_models(models, Xy, parameter_ranges)
		print("#" * 20)
		for model_name, train_score, test_score, train_std_dev, test_std_dev in zip(model_names, train_scores, test_scores, train_std_devs, test_std_devs):
			print("%s: Train MAE = %f +- %f Test MAE = %f +- %f" % (model_name, train_score, train_std_dev, test_score, test_std_dev))
		print("#" * 20)
