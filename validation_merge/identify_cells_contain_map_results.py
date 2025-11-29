#!/usr/bin/env python
# Filename: identify_cells_contain_map_results.py
"""
introduction: to identify cells or grids, likey contain true positives for validation.

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 04 September, 2025
"""

import os,sys
import time
from optparse import OptionParser

import numpy as np

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

import basic_src.io_function as io_function
import basic_src.basic as basic
import datasets.vector_gpd as vector_gpd

from datetime import datetime

import geopandas as gpd
from collections import Counter
import csv

def read_numpy_from_file(npy_file_list,col_name):
    for npy in npy_file_list:
        if os.path.basename(npy) ==  f'{col_name}.npy':
            return np.load(npy)

    raise IOError(f'{col_name}.npy does not exist in the list: {npy_file_list}')

def read_count_area_array(grid_gpd,count_columns,area_columns, npy_file_list = None):
    if npy_file_list is not None:
        count_array_list =  [read_numpy_from_file(npy_file_list, item) for item in count_columns ]
        area_array_list =  [read_numpy_from_file(npy_file_list, item) for item in area_columns ]
    else:
        count_array_list = [np.array(grid_gpd[item]) for item in count_columns ]
        area_array_list = [ np.array(grid_gpd[item]) for item in area_columns ]

    count_array_2d = np.vstack(count_array_list)
    area_array_2d = np.vstack(area_array_list)
    return count_array_2d, area_array_2d

def calculate_s2_detection_occur_trend(grid_gpd,npy_file_list = None):
    # s2 count
    s2_area_columns = ['s2_2018_A', 's2_2019_A', 's2_2020_A', 's2_2021_A', 's2_2022_A', 's2_2023_A', 's2_2024_A']
    s2_count_columns = ['s2_2018_C', 's2_2019_C', 's2_2020_C', 's2_2021_C', 's2_2022_C', 's2_2023_C', 's2_2024_C']

    count_array_2d, area_array_2d = read_count_area_array(grid_gpd, s2_count_columns, s2_area_columns,
                                                          npy_file_list=npy_file_list)

    print('count_array_2d', count_array_2d.shape)
    print('area_array_2d', area_array_2d.shape)

    # calcuate the occurence of detection in each year
    # if 1 or more there, count a 1, otherwise, as zero
    count_array_2d_sum = np.sum(count_array_2d, axis=0)  # sum across different years
    count_array_2d_binary = (count_array_2d > 0).astype(int)
    count_array_2d_binary_sum = np.sum(count_array_2d_binary, axis=0)  # sum across different years
    print('count_array_2d_binary_sum', count_array_2d_binary_sum.shape, np.min(count_array_2d_binary_sum),
          np.max(count_array_2d_binary_sum), np.mean(count_array_2d_binary_sum))


    # selection based on area changes
    area_array_2d_sum = np.sum(area_array_2d, axis=0)  # sum across different years
    x = np.arange(area_array_2d.shape[0])  # shape (7,)
    # Calculate slopes (trend) per column, # trends[i] is the trend (slope) for column i
    trends = np.polyfit(x, area_array_2d, deg=1)[0]  # shape (13304783,)
    print('trends', trends.shape, trends)

    return count_array_2d_binary_sum, trends, count_array_2d_sum, area_array_2d_sum


def find_grid_base_on_s2_results(grid_gpd, annual_count_thr=5, area_trend_thr=0.01, npy_file_list = None):

    count_array_2d_binary_sum, trends,count_array_2d_sum, area_array_2d_sum \
        = calculate_s2_detection_occur_trend(grid_gpd, npy_file_list=npy_file_list)

    # selection based on the count
    b_select_on_count = count_array_2d_binary_sum  >= annual_count_thr
    print('b_select_on_count:', b_select_on_count.shape, b_select_on_count)

    b_select_on_area_trend = trends >= area_trend_thr
    print('b_select_on_area_trend:', b_select_on_area_trend.shape, b_select_on_area_trend)

    # conduct the select
    select_idx = np.logical_and(b_select_on_count, b_select_on_area_trend)
    if grid_gpd is None:
        print('For testing, grid_gpd is None')
        return None
    # grid_gpd_sel = grid_gpd[select_idx]
    # grid_gpd_sel['s2_occur'] = count_array_2d_binary_sum[select_idx]
    # grid_gpd_sel['s2_area_trend'] = trends[select_idx]
    s2_occur = count_array_2d_binary_sum[select_idx]
    s2_area_trend = trends[select_idx]
    s2_count_sum = count_array_2d_sum[select_idx]
    s2_area_sum = area_array_2d_sum[select_idx]
    basic.outputlogMessage(f'Select {len(s2_occur)} cells from {len(select_idx)} based on S2 mapping results')
    return select_idx, s2_occur, s2_area_trend, s2_count_sum, s2_area_sum


def find_grid_base_on_DEM_results(grid_gpd, npy_file_list = None):

    # DEM
    dem_area_columns = ['samElev_A', 'comImg_A']
    dem_count_columns = ['samElev_C', 'comImg_C']

    count_array_2d, area_array_2d = read_count_area_array(grid_gpd, dem_count_columns, dem_area_columns,
                                                          npy_file_list=npy_file_list)

    print('count_array_2d', count_array_2d.shape, np.min(count_array_2d, axis=1), np.max(count_array_2d,axis=1), np.mean(count_array_2d,axis=1) )
    print('area_array_2d', area_array_2d.shape, np.min(area_array_2d,axis=1), np.max(area_array_2d,axis=1), np.mean(area_array_2d,axis=1))




def test_find_grid_base_on_s2_results():
    npy_file_list = io_function.get_file_list_by_ext('.npy','./', bsub_folder=False)
    find_grid_base_on_s2_results(None, npy_file_list=npy_file_list)

def test_find_grid_base_on_DEM_results():
    npy_file_list = io_function.get_file_list_by_ext('.npy', './', bsub_folder=False)
    find_grid_base_on_DEM_results(None, npy_file_list=npy_file_list)


def merge_validation_from_users(validate_json_list,save_path="valid_res_dict.json"):
    valid_res_dict = {}
    for v_file in validate_json_list:
        data_dict = io_function.read_dict_from_txt_json(v_file)
        res_list = []
        h3_id = data_dict['h3ID']
        for key in data_dict.keys():
            if key == "h3ID":
                continue
            res_list.append(data_dict[key]['ValidateResult'])

        if len(res_list) == 1:
            valid_res_dict[h3_id] = res_list[0]
        elif len(res_list) == 2:
            if len(set(res_list)) != 1:
                basic.outputlogMessage(f'Warning, the validation results in {v_file} from two contributors disagree, choose the first one')
            valid_res_dict[h3_id] = res_list[0]
        elif len(res_list) > 2:
            if len(set(res_list)) != 1:
                basic.outputlogMessage(f'Warning, the validation results in {v_file} from {len(res_list)} contributors disagree, '
                                       f'choose the common')
            # Count the occurrences of each element
            counts = Counter(res_list)
            value, frequency = counts.most_common(1)[0]
            valid_res_dict[h3_id] = value
        else:
            raise ValueError(f'No validation results in {v_file}')

    io_function.save_dict_to_txt_json(save_path,valid_res_dict)
    return valid_res_dict

def merge_validation_from_users_weight(validate_json_list,user_weight='valid_user_weight.csv', save_path="valid_res_dict.json"):
    valid_res_dict = {}
    if os.path.isfile(user_weight) is False:
        raise IOError(f'{user_weight} not exists')

    # read user weight
    user_weight_dict = {}
    all_user_input_dict = {}
    with open(user_weight, mode='r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # print(row)
            user_weight_dict[row['user_name']] = float(row['weight'])

    io_function.save_dict_to_txt_json(io_function.get_name_no_ext(user_weight)+'.json', user_weight_dict)
    # print(user_weight_dict)
    # sys.exit(0)


    for v_file in validate_json_list:
        data_dict = io_function.read_dict_from_txt_json(v_file)
        user_input_dict = {}
        h3_id = data_dict['h3ID']
        for key in data_dict.keys():    # key is the user_name
            if key == "h3ID":
                continue
            user_input_dict.setdefault(data_dict[key]['ValidateResult'],[]).append({key:user_weight_dict[key]})

        # print(user_input_dict)
        # print(len(user_input_dict))

        if len(user_input_dict) == 1:
            valid_res_dict[h3_id] = list(user_input_dict.keys())[0] # if only one results, or multple result but agree, then
        elif len(user_input_dict) > 1:
            # calculate the result based on weight
            max_val_weight = 0
            value = 'TP'
            for val_key in user_input_dict:
                val_sum_weight = sum([ list(i_dict.values())[0] for i_dict in user_input_dict[val_key]])
                if val_sum_weight > max_val_weight:
                    max_val_weight = val_sum_weight
                    value = val_key
            valid_res_dict[h3_id] =  value
        else:
            raise ValueError(f'No validation results in {v_file}')

        all_user_input_dict[h3_id] = user_input_dict

    io_function.save_dict_to_txt_json(save_path,valid_res_dict)
    all_user_input_dict_save = io_function.get_name_by_adding_tail(save_path,'allUsersInput')
    io_function.save_dict_to_txt_json(all_user_input_dict_save,all_user_input_dict)
    return valid_res_dict

def load_training_data_from_validate_jsons(validate_json_list,save_path="valid_res_dict.json"):

    # merge validation results from different user by rules
    # valid_res_dict = merge_validation_from_users(validate_json_list,save_path=save_path)
    valid_res_dict = merge_validation_from_users_weight(validate_json_list,save_path=save_path)

    # only keep TP and FP, and convert them into 1 and 0
    valid_res_dict_int_labels = {}
    for key, value  in valid_res_dict.items():
        if value == 'TP':
            valid_res_dict_int_labels[key] = 1
        elif value == 'FP':
            valid_res_dict_int_labels[key] = 0
        else:
            pass

    return valid_res_dict_int_labels

def extract_columns_as_dict(grid_gpd):

    column_pre_names = ['s2','samElev','comImg','susce']
    for col_name in grid_gpd.columns:
        # col_name start with one of the pre_name
        if not col_name.startswith(column_pre_names):
            continue

def prepare_training_data(grid_gpd, validate_res_dict,feature_pre_names,id_col):

    # read data into numpy array
    h3_id_np = np.array(grid_gpd[id_col]).astype(str)
    # print(h3_id_np.dtype)# .astype(str)
    h3_id_np_uniq, h3_id_inv = np.unique(h3_id_np, return_inverse=True)
    if len(h3_id_np) != len(h3_id_np_uniq):
        raise ValueError('there are repeat element in h3_id in numpy array')

    features_np_list = []
    feature_cols = []
    for col_name in grid_gpd.columns:
        # col_name start with one of the pre_name
        if not col_name.startswith(tuple(feature_pre_names)):
            continue
        feature_cols.append(col_name)
        data_np = np.array(grid_gpd[col_name])
        if col_name.startswith('susce'):
            # Replace NaN values with zero
            data_np[np.isnan(data_np)] = 0
        features_np_list.append(data_np)
    features_np_2d = np.stack(features_np_list,axis=1)  # shape (n_samples, n_features)
    print('features_np_2d shape:', features_np_2d.shape)

    # prepare training data
    # using numpy, not for-loop
    # compare the results with produced by for-loop, they are the same but in different orders
    label_keys = np.array(list(validate_res_dict.keys()), dtype=str)
    label_keys_uniq = np.unique(label_keys)
    if len(label_keys) != len(label_keys_uniq):
        raise ValueError('there are repeat elements of h3_id in validate_res_dict')
    # intersect1d will sort h3_id_np_uniq and label_keys
    overlap, idx_h3_id, idx_l = np.intersect1d(h3_id_np_uniq, label_keys,return_indices=True)
    fill_value = -9999
    labels_for_unique = np.full(h3_id_np_uniq.shape, fill_value, dtype=int)
    # print(np.array([int(validate_res_dict[k]) for k in label_keys[idx_l]], dtype=int))
    labels_for_unique[idx_h3_id] = np.array([int(validate_res_dict[k]) for k in label_keys[idx_l]], dtype=int)
    # print(labels_for_unique,labels_for_unique.shape)
    row_labels = labels_for_unique[h3_id_inv]
    mask = row_labels != fill_value
    train_features_2d = features_np_2d[mask,:]
    train_labels_1d = row_labels[mask]
    train_h3_ids = h3_id_np[mask]


    # # # using for-loop
    # train_features = []
    # train_labels = []
    # train_h3_ids = []
    # for h_id in validate_res_dict.keys():
    #     row_idx = np.where(h3_id_np == h_id)[0]    #
    #     if row_idx.size == 0:
    #     # skip if not found
    #         continue
    #     # print(row_idx)
    #     train_h3_ids.append(h_id)
    #     row_idx_int = int(row_idx[0])
    #     train_features.append(features_np_2d[row_idx_int,:])
    #     # ensure 2D row shape
    #     train_labels.append(validate_res_dict[h_id])
    #     # # print out for checking
    #     # if h_id=='8802f2d461fffff':
    #     #     print(f"{h_id}: label: {validate_res_dict[h_id]}, features {features_np_2d[row_idx_int,:]}")
    # train_features_2d = np.stack(train_features)
    # train_labels_1d = np.array(train_labels)
    # train_h3_ids = np.array(train_h3_ids)


    print('train_features_2d shape:', train_features_2d.shape)
    print('train_labels_1d shape:', train_labels_1d.shape)
    print('train_h3_ids shape:', train_h3_ids.shape)
    # save for checking
    # np.save("train_features_2d.npy",train_features_2d)
    # np.save("train_labels_1d.npy",train_labels_1d)
    # np.save("train_h3_ids.npy",train_h3_ids)

    #  X, y, X_all, ids_all, feature_cols
    return train_features_2d, train_labels_1d, features_np_2d, h3_id_np, feature_cols



def test_load_training_data_from_validate_jsons():
    # data_dir = os.path.expanduser('~/Data/rts_ArcticDEM_mapping/validation/select_by_s2_result_png')
    # json_list = io_function.get_file_list_by_pattern(data_dir,'*/validated*.json')

    data_dir = os.path.expanduser('~/Data/rts_ArcticDEM_mapping/validation/validated_json/valid_json_files_20251128')
    json_list = io_function.get_file_list_by_pattern(data_dir,'validated_*.json')

    save_valid_res_dict = 'test_valid_res_dict.json'
    load_training_data_from_validate_jsons(json_list, save_path=save_valid_res_dict)
    pass

def train_randomforest_with_hyperpara_search(X, y, out_json_path="rf_hyperparam_search_results.json"):
    """
    Train a RandomForest with hyperparameter search and save results.
    - Uses RandomizedSearchCV with StratifiedKFold and F1 scoring.
    - Saves all candidates' params and scores plus the best params/score to JSON.
    Returns:
        best_model: fitted RandomForestClassifier with best params
        cv_f1_mean: float, best cross-validated F1 score
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
    from sklearn.metrics import make_scorer, f1_score

    # Base model (some params tuned)
    base_rf = RandomForestClassifier(
        n_estimators=300,    # will be overridden by search
        max_depth=None,      # tuned
        min_samples_leaf=1,  # tuned
        max_features="sqrt", # tuned
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )

    # CV folds based on minority class count
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    min_class = max(1, min(pos, neg))
    n_splits = max(2, min(5, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Search space
    param_dist = {
        "n_estimators": [200, 300, 400, 600],
        "max_depth": [None, 6, 10, 16, 24],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.5, 0.7, None],
        "min_samples_split": [2, 4, 8, 16],
        "bootstrap": [True, False],
    }

    scorer = make_scorer(f1_score, average="binary")

    total_grid = int(np.prod([len(v) for v in param_dist.values()]))
    n_iter = min(30, total_grid)

    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scorer,
        n_jobs=-1,
        cv=cv,
        verbose=0,
        random_state=42,
        refit=True,
        return_train_score=False,
    )

    search.fit(X, y)
    best_model = search.best_estimator_
    cv_f1_mean = float(search.best_score_) if search.best_score_ is not None else None

    # Collect full results
    cv_results = search.cv_results_
    # Build a compact list of all tried params with their mean/std test scores
    all_trials = []
    for i in range(len(cv_results["params"])):
        all_trials.append({
            "rank_test_score": int(cv_results["rank_test_score"][i]),
            "mean_test_score": float(cv_results["mean_test_score"][i]),
            "std_test_score": float(cv_results["std_test_score"][i]),
            "params": cv_results["params"][i],
        })

    out_payload = {
        "cv_n_splits": n_splits,
        "scoring": "f1",
        "n_iter": int(n_iter),
        "best_score": cv_f1_mean,
        "best_params": search.best_params_,
        "all_trials": all_trials,
    }

    # Save to JSON
    io_function.save_dict_to_txt_json(out_json_path,out_payload)

    return best_model, cv_f1_mean

def rf_feature_importance(model, feature_cols, top_k=None):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    if top_k is not None:
        order = order[:top_k]
    out = {feature_cols[i]: float(importances[i]) for i in order}
    return out


def rf_permutation_importance(model, X, y, feature_cols, n_repeats=10, random_state=42):
    # Use oob_score_ if enabled, otherwise compute on the training set
    from sklearn.inspection import permutation_importance

    r = permutation_importance(
        model, X, y,
        scoring="f1",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )
    means = r.importances_mean
    stds = r.importances_std
    order = np.argsort(means)[::-1]
    # out = {feature_cols[i]: {"mean": float(means[i]), "std": float(stds[i])} for i in order}
    out = {feature_cols[i]:  float(means[i]) for i in order}
    return out

def rf_shap_importance(model, X, feature_cols, sample_size=2000, random_state=42, positive_class=1):
    """
        Returns a dict: {feature_name: mean_abs_shap} ordered by descending mean |SHAP|.
        For binary classification, uses SHAP values for the positive class by default.
    """

    import shap

    if X.shape[0] > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X.shape[0], size=sample_size, replace=False)
        X_use = X[idx]
    else:
        X_use = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_use)

    # Select class if classification returns a list
    if isinstance(shap_values, list):
        classes_ = getattr(model, "classes_", None)
        if positive_class is None:
            sv = shap_values[-1]
        elif classes_ is not None and positive_class in classes_:
            class_idx = int(np.where(classes_ == positive_class)[0][0])
            sv = shap_values[class_idx]
        else:
            sv = shap_values[-1]
    else:
        sv = shap_values

    # Ensure sv is at least 2D: (n_samples, n_features, [maybe extra axes])
    sv = np.array(sv)

    # If sv has more than 2 dims, collapse all axes except the last feature axis and the sample axis.
    # Common shapes:
    # - (n_samples, n_features) -> keep
    # - (n_samples, n_features, K) -> average over axis=2
    # - (K, n_samples, n_features) -> average over axis=0
    if sv.ndim == 3:
        # Decide which axis is features: usually the last axis is features
        # Check which axis equals n_features inferred from feature_cols length
        n_features = len(feature_cols)
        if sv.shape[-1] == n_features:
            # (n_samples, n_features, K) -> mean over K
            sv = sv.mean(axis=2)
        elif sv.shape[1] == n_features:
            # (K, n_features, n_samples) or (K, n_samples, n_features)
            # Most common is (K, n_samples, n_features) for per-tree outputs
            if sv.shape[-1] == n_features:
                # (K, n_samples, n_features) -> mean over K
                sv = sv.mean(axis=0)
            else:
                # (K, n_features, n_samples) -> move features to last, then mean K
                sv = sv.transpose(0, 2, 1).mean(axis=0)
        else:
            # Fallback: mean across the first axis
            sv = sv.mean(axis=0)

    elif sv.ndim > 3:
        # Collapse all leading axes except the last two (samples, features)
        while sv.ndim > 2:
            # Average the first axis until we get 2D
            sv = sv.mean(axis=0)

    # Now sv should be (n_samples, n_features)
    if sv.ndim != 2 or sv.shape[1] != len(feature_cols):
        raise ValueError(f"Unexpected SHAP values shape {sv.shape}; expected (_, {len(feature_cols)})")

    mean_abs = np.abs(sv).mean(axis=0)  # shape: (n_features,)
    order = np.argsort(mean_abs)[::-1].astype(int)

    # print('order:', order)
    out = {feature_cols[i]: float(mean_abs[i]) for i in order}
    return out


def auto_find_positive_grids(grid_gpd,validate_json_list, save_path, proba_thr=0.5):
    # using machine learning algorithm to find grid that likely contains thaw targets

    if 's2_occur' in grid_gpd.columns and 's2_area_trend' in grid_gpd.columns:
        pass
    else:
        count_array_2d_binary_sum, trends,count_array_2d_sum, area_array_2d_sum \
        = calculate_s2_detection_occur_trend(grid_gpd, npy_file_list=None)
        grid_gpd['s2_occur'] = count_array_2d_binary_sum
        grid_gpd['s2_area_trend'] = trends

    save_file_basename = io_function.get_name_no_ext(save_path)
    validate_json_list_file = save_file_basename + '_valid_res_dict.json'
    validate_res_dict = load_training_data_from_validate_jsons(validate_json_list, save_path=validate_json_list_file)
    if len(validate_res_dict) < 10:
        raise ValueError(f'Only {len(validate_res_dict)} labeled samples, not enough to train a model')

    column_pre_names = ['s2', 'samElev', 'comImg', 'susce']
    X, y, X_all, ids_all, feature_cols = prepare_training_data(grid_gpd, validate_res_dict, column_pre_names,'h3_id_8')
    basic.outputlogMessage('completed: preparing training data')

    # 3) Train model (Random Forest preferred)
    rf_hp_search_save_path = save_file_basename + "_rf_hyperparam_search_results.json"
    rf, cv_f1_mean = train_randomforest_with_hyperpara_search(X,y,out_json_path=rf_hp_search_save_path)
    basic.outputlogMessage('completed: training with hyper-parameters searching')

    ############ checking the importance of each feature #########
    # 1) Impurity-based
    impurity_rank = rf_feature_importance(rf, feature_cols, top_k=20)
    io_function.save_dict_to_txt_json(save_file_basename+'_feature_importance_rank_impurity.json',impurity_rank)
    # 2) Permutation-based (slower)
    perm_rank = rf_permutation_importance(rf, X, y, feature_cols, n_repeats=10)
    io_function.save_dict_to_txt_json(save_file_basename+'_feature_importance_rank_permutation.json', perm_rank)
    # 3) SHAP-based (slower)
    shap_rank = rf_shap_importance(rf, X, feature_cols,sample_size=5000)
    # print(shap_rank)
    io_function.save_dict_to_txt_json(save_file_basename+'_feature_importance_rank_shap.json', shap_rank)
    basic.outputlogMessage('completed: sorting feature importance')
    ######################################################################

    # 4) Predict probabilities for all rows
    proba = rf.predict_proba(X_all)[:, 1]
    basic.outputlogMessage('completed: prediction')

    # 6) Return outputs similar to previous contract
    info = {
        "n_labeled": int(y.size),
        "class_balance": {"neg": int((y == 0).sum()), "pos": int((y == 1).sum())},
        "feature_cols": feature_cols,
        "model_type": "RandomForestClassifier",
        "cv_f1_mean": cv_f1_mean,
    }
    io_function.save_dict_to_txt_json(save_file_basename+'_random_forest_info.json', info)

    # save to file
    threoshold_list = [proba_thr] if isinstance(proba_thr,float) else proba_thr
    for threoshold in threoshold_list:

        if len(threoshold_list) > 1:
            save_path_tmp = io_function.get_name_by_adding_tail(save_path,f'{threoshold}')
        else:
            save_path_tmp = save_path

        pred_binary = (proba >= threoshold).astype(int)
        select_idx = proba >= threoshold

        # Create/assign columns
        grid_gpd["pred_proba_TP"] = proba
        grid_gpd["pred_label"] = np.where(pred_binary == 1, "TP", "FP")

        select_grid_gpd = grid_gpd[select_idx]
        select_grid_gpd.to_file(save_path_tmp)
        basic.outputlogMessage(f'Saved {len(select_grid_gpd)} selected cells to {save_path_tmp}')



def identify_cells_contain_true_results(grid_gpd, save_path, train_data_dir=None,
                            train_file_pattern='*/validated*.json', method='s2_area_count'):

    if method.lower() == 's2_area_count':
        # select based on sentinel-2
        select_idx, s2_occur, s2_area_trend, s2_count_sum, s2_area_sum =\
            find_grid_base_on_s2_results(grid_gpd)
        select_grid_gpd = grid_gpd[select_idx]
        select_grid_gpd['s2_occur'] = s2_occur
        select_grid_gpd['s2_area_trend'] = s2_area_trend
        select_grid_gpd['s2_count_sum'] = s2_count_sum
        select_grid_gpd['s2_area_sum'] = s2_area_sum

        select_grid_gpd.to_file(save_path)
    elif method.lower() == 'random_forest':
        if train_data_dir is None:
            raise ValueError("train_data_dir is not set")
        validate_json_list = io_function.get_file_list_by_pattern(train_data_dir,train_file_pattern)
        if len(validate_json_list) < 1:
            raise ValueError(f'No validated*.json files in {train_data_dir}')
        basic.outputlogMessage(f'Found {len(validate_json_list)} validated*.json files in {train_data_dir}')

        prob_thr = [0.5,0.6,0.7,0.8,0.9]
        auto_find_positive_grids(grid_gpd, validate_json_list, save_path,proba_thr=prob_thr)

    else:
        raise ValueError('Unknown method for identifying cells')

def main(options, args):
    grid_path = args[0]
    save_path = options.save_path
    train_data_dir = options.train_data_dir
    method = options.method
    train_file_pattern = options.train_file_pattern


    t0 = time.time()
    grid_gpd = gpd.read_file(grid_path)
    t1 = time.time()
    print(f'Loaded grid vector file, containing {len(grid_gpd)} cells, {len(grid_gpd.columns)} columns, cost {t1-t0} seconds')
    print('column names:', grid_gpd.columns.to_list())

    identify_cells_contain_true_results(grid_gpd, save_path, train_data_dir=train_data_dir, train_file_pattern=train_file_pattern,method=method)



    pass


if __name__ == '__main__':

    # test_find_grid_base_on_s2_results()
    # test_find_grid_base_on_DEM_results()
    test_load_training_data_from_validate_jsons()
    sys.exit(0)

    usage = "usage: %prog [options] grid_vector "
    parser = OptionParser(usage=usage, version="1.0 2025-9-4")
    parser.description = 'Introduction: identify grid (cells) that likely contain true mapping results '

    parser.add_option("-s", "--save_path",
                      action="store", dest="save_path",default="select_cells.gpkg",
                      help="the save path ")

    parser.add_option("-i", "--input_txt",
                      action="store", dest="input_txt",
                      help="the input txt contain column name and vector path (column_name, vector_path)")

    parser.add_option("-m", "--method",
                      action="store", dest="method",
                      help="the method to identify cell containing positive results, including: s2_area_count, random_forest, etc")

    parser.add_option("-d", "--train_data_dir",
                      action="store", dest="train_data_dir",
                      help="the the folder containing the validated*.json, specific as follows ")

    parser.add_option("-p", "--train_file_pattern",
                      action="store", dest="train_file_pattern", default="8*/validated*.json",
                      help="the train file patterns ")


# train_data_dir

    (options, args) = parser.parse_args()
    # print(options.no_label_image)
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)