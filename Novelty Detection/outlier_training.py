import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import argparse
from datetime import timedelta

import sys
sys.path.append('..')

import warnings
warnings.filterwarnings('ignore')
import pickle

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model_type', type=str, help='OCSVM or Isolation Forest')
    parser.add_argument('--window_size', type=int, default=2, help="Size of the window used to train")
    parser.add_argument('--contamination', default=0.01, help="Outliers on training data")
    parser.add_argument('--training_method', type=str, default= 'cluster_0', help='All samples or only cluster 0 samples')
    
    args = parser.parse_args()
    
    print(f"Running script with: {args.model_type, args.window_size, args.contamination, args.training_method}")
    return args

def start_of_month(date):
    return date.replace(day=1)

def end_of_month(date):
    next_month = date.replace(day=28) + timedelta(days=4)  # this will never fail
    return next_month - timedelta(days=next_month.day)

if __name__ == '__main__':

    # Parse arguments
    args = parse_arguments()
    model_type = args.model_type
    window_size = pd.DateOffset(months=args.window_size)
    contamination = args.contamination if args.contamination == 'inhereted' else float(args.contamination)
    training_method = args.training_method

    # Load data
        # Labels
    MALDI_type = 'peaks'
    with open(f'../../MALDI data/{MALDI_type}_info_dict.pkl', 'rb') as f:
        labels_dict = pickle.load(f)

        # Kernel
    with open(f'../../MALDI data/{MALDI_type}_kernels.pkl', 'rb') as f:
        kernel_dict = pickle.load(f)
    
    bins = 'Ct. bins'
    labels_df = labels_dict[bins]
    kernel_matrix = kernel_dict[bins]

    labels_df = labels_df.reset_index()
    
    # Configure constant sliding window
    step_size = pd.DateOffset(months=1)
    start_date = labels_df['F REGISTRO'].min()
    end_date = labels_df['F REGISTRO'].max()

    # Train - test loop
    current_start = start_of_month(start_date)
    while current_start + window_size < end_date:
        train_start = current_start
        train_end = end_of_month(current_start + window_size - pd.DateOffset(days=1))
        test_start = train_end + timedelta(days=1)
        test_end = end_of_month(test_start + step_size - pd.DateOffset(days=1))

        # Select training and testing indices based on training method
        if training_method == 'cluster_0':
            train_indices = labels_df[(labels_df['F REGISTRO'] >= train_start) & (labels_df['F REGISTRO'] < train_end) & (labels_df['CLUSTER'] == 0)].index
        elif training_method == 'all_samples':
            train_indices = labels_df[(labels_df['F REGISTRO'] >= train_start) & (labels_df['F REGISTRO'] < train_end)].index

            if contamination == 'inhereted':
                num_samples_train_outlier = len(labels_df[(labels_df['F REGISTRO'] >= train_start) & (labels_df['F REGISTRO'] <= train_end) & (labels_df['CLUSTER'] != 0)].index)
                num_samples_train = len(train_indices)
                contamination = num_samples_train_outlier / num_samples_train

        test_indices = labels_df[(labels_df['F REGISTRO'] >= test_start) & (labels_df['F REGISTRO'] < test_end)].index

        # Prepare training and testing data
        if len(train_indices) > 0 and len(test_indices) > 0:
            X_train = kernel_matrix[np.ix_(train_indices, train_indices)]
            X_test = kernel_matrix[np.ix_(test_indices, train_indices)]  # use training indices for kernel alignment

            # Train the model
            print(f"Training with {X_train.shape[0]} samples...")
            if model_type == 'ocsvm':
                model = OneClassSVM(kernel='precomputed', nu=contamination)
            elif model_type == 'isolation_forest':
                model = IsolationForest(contamination=contamination, random_state=10, bootstrap=True)

            model.fit(X_train)

            # Predict on the test set
            print(f"Testing with {X_test.shape[0]} samples...")
            y_pred = model.predict(X_test)

            # Save results for evaluation
            results_dict = {'train_start': train_start, 
                            'train_end': train_end, 
                            'test_start': test_start, 
                            'test_end': test_end, 
                            'X_train': X_train, 
                            'X_test': X_test, 
                            'model_type': model_type,
                            'train_indices': train_indices, 
                            'test_indices': test_indices, 
                            'pred': y_pred, 
                            'model': model,
                            'decision_function': model.decision_function(X_test)
                            }
            
            start_date_str = str(test_start.date()).replace("-", "_")
            end_date_str = str(test_end.date()).replace("-", "_")
            with open(f'../../MALDI data/novelty/preds_{start_date_str}_{end_date_str}_{model_type}.pkl', 'wb') as f:
                pickle.dump(results_dict, f)

            # Evaluate performance (example: calculate the ratio of -1s as outliers)
            outliers = np.sum(y_pred == -1)
            print(f"Training period: {train_start.date()} to {train_end.date()}")
            print(f"Testing period: {test_start.date()} to {test_end.date()}")
            print(f"Outliers detected: {outliers} out of {len(y_pred)} samples")
            print("="*50)

        current_start += step_size

