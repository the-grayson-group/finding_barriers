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
from reaction_repr import get_michael_features, get_vaska_features
from model_evaluation import test_models

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

	datasets = [get_michael_features("barrier_feature_groups.csv", "../reactant_structures/"), get_vaska_features("vaska_features_properties_smiles_filenames.csv")]
	model_names = ["LR", "RR", "RFR", "GBR", "KRR", "GPR"]
	for name, Xy in zip(("Michael Features", "Vaska Features"), datasets):
		print(name)
		train_scores, test_scores, train_std_devs, test_std_devs = test_models(models, Xy, parameter_ranges)
		print("#" * 20)
		for model_name, train_score, test_score, train_std_dev, test_std_dev in zip(model_names, train_scores, test_scores, train_std_devs, test_std_devs):
			print("%s: Train MAE = %f +- %f Test MAE = %f +- %f" % (model_name, train_score, train_std_dev, test_score, test_std_dev))
		print("#" * 20)
