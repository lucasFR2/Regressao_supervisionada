import os

import json

import math

import random

from ml_from_scratch import (

    load_and_preprocess_scratch,

    train_test_split,

    StandardScalerScratch,

    LinearRegressionGD,

    DecisionTreeRegressorScratch,

    RandomForestRegressorScratch,

    calculate_metrics

)



def compute_histogram(data, bins=10):

    """Calcula a distribuição em bins para um histograma."""

    if not data:

        return [], []

    min_val, max_val = min(data), max(data)

                                    

    if min_val == max_val:

        return [min_val], [len(data)]

        

    bin_width = (max_val - min_val) / bins

    bin_edges = [min_val + i * bin_width for i in range(bins + 1)]

    counts = [0] * bins

    

    for val in data:

        for i in range(bins):

            if bin_edges[i] <= val < bin_edges[i+1]:

                counts[i] += 1

                break

        else:

            if val == max_val:

                counts[-1] += 1

                

                              

    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(bins)]

    return bin_labels, counts



def compute_correlation_matrix(X, col_names):

    """Calcula a matriz de correlação de Pearson para as colunas numéricas."""

    n_samples = len(X)

    n_features = len(col_names)

    

                                                        

    means = []

    stds = []

    for j in range(n_features):

        col = [X[i][j] for i in range(n_samples)]

        mean = sum(col) / n_samples

        variance = sum((x - mean) ** 2 for x in col) / n_samples

        std = math.sqrt(variance) if variance > 0 else 1.0

        means.append(mean)

        stds.append(std)

        

                                   

    corr_matrix = []

    for j1 in range(n_features):

        row = []

        for j2 in range(n_features):

            col1 = [X[i][j1] for i in range(n_samples)]

            col2 = [X[i][j2] for i in range(n_samples)]

            

                         

            cov = sum((col1[i] - means[j1]) * (col2[i] - means[j2]) for i in range(n_samples)) / n_samples

                                   

            corr = cov / (stds[j1] * stds[j2]) if (stds[j1] * stds[j2]) > 0 else 0.0

            row.append(corr)

        corr_matrix.append(row)

        

    return corr_matrix



def main():

    print("Iniciando pipeline de análise em Python Puro (Sem dependências binárias)...")

    

                                 

    mat_csv = os.path.join("data", "student-mat.csv")

    por_csv = os.path.join("data", "student-por.csv")

    

    datasets = {

        "Matematica": mat_csv,

        "Portugues": por_csv

    }

    

    all_results = {}

    

    for subject, file_path in datasets.items():

        print(f"\nProcessando disciplina: {subject}...")

        

                                     

        X, y, feature_names, num_cols, cat_cols = load_and_preprocess_scratch(file_path)

        

                                        

                               

        hist_labels, hist_counts = compute_histogram(y, bins=10)

        

                                                                

                                                                        

        num_indices = list(range(len(num_cols)))

        X_num_only = [[row[idx] for idx in num_indices] for row in X]

        corr_matrix = compute_correlation_matrix(X_num_only, num_cols)

        

                                              

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        

                                                                                    

        scaler = StandardScalerScratch()

                                                                 

        X_train_num = [[row[idx] for idx in num_indices] for row in X_train]

        X_test_num = [[row[idx] for idx in num_indices] for row in X_test]

        

        scaler.fit(X_train_num)

        X_train_num_scaled = scaler.transform(X_train_num)

        X_test_num_scaled = scaler.transform(X_test_num)

        

                                                                         

        X_train_scaled = []

        for i in range(len(X_train)):

            new_row = X_train_num_scaled[i] + X_train[i][len(num_cols):]

            X_train_scaled.append(new_row)

            

        X_test_scaled = []

        for i in range(len(X_test)):

            new_row = X_test_num_scaled[i] + X_test[i][len(num_cols):]

            X_test_scaled.append(new_row)

            

                      

        print("Treinando Regressão Linear...")

        lr_model = LinearRegressionGD(lr=0.05, epochs=1500)

        lr_model.fit(X_train_scaled, y_train)

        y_train_pred_lr = lr_model.predict(X_train_scaled)

        y_test_pred_lr = lr_model.predict(X_test_scaled)

        metrics_lr = calculate_metrics(y_test, y_test_pred_lr)

        metrics_lr_train = calculate_metrics(y_train, y_train_pred_lr)

        

        print("Treinando Árvore de Decisão...")

        dt_model = DecisionTreeRegressorScratch(max_depth=5, min_samples_split=5)

        dt_model.fit(X_train_scaled, y_train)

        y_train_pred_dt = dt_model.predict(X_train_scaled)

        y_test_pred_dt = dt_model.predict(X_test_scaled)

        metrics_dt = calculate_metrics(y_test, y_test_pred_dt)

        metrics_dt_train = calculate_metrics(y_train, y_train_pred_dt)

        

        print("Treinando Random Forest...")

        rf_model = RandomForestRegressorScratch(n_estimators=30, max_depth=6, min_samples_split=5, random_state=42)

        rf_model.fit(X_train_scaled, y_train)

        y_train_pred_rf = rf_model.predict(X_train_scaled)

        y_test_pred_rf = rf_model.predict(X_test_scaled)

        metrics_rf = calculate_metrics(y_test, y_test_pred_rf)

        metrics_rf_train = calculate_metrics(y_train, y_train_pred_rf)

        

                                                      

        rf_importances = rf_model.get_feature_importances(len(feature_names))

        feat_imp_pairs = sorted(zip(feature_names, rf_importances), key=lambda x: x[1], reverse=True)

        top_importances = [{"feature": f, "importance": imp} for f, imp in feat_imp_pairs[:10]]

        

                                          

        lr_coefs = lr_model.weights

        lr_coef_pairs = sorted(zip(feature_names, lr_coefs), key=lambda x: abs(x[1]), reverse=True)

        top_coefs = [{"feature": f, "coefficient": coef} for f, coef in lr_coef_pairs[:10]]

        

                                                                                                                            

                                             

        random.seed(42)

        sample_indices = random.sample(range(len(y_test)), min(80, len(y_test)))

        

        scatter_data = {

            "y_test_subset": [y_test[idx] for idx in sample_indices],

            "predictions_lr": [y_test_pred_lr[idx] for idx in sample_indices],

            "predictions_dt": [y_test_pred_dt[idx] for idx in sample_indices],

            "predictions_rf": [y_test_pred_rf[idx] for idx in sample_indices]

        }

        

                                                              

        all_results[subject] = {

            "sample_counts": {

                "total": len(y),

                "train": len(y_train),

                "test": len(y_test)

            },

            "g3_distribution": {

                "labels": hist_labels,

                "counts": hist_counts

            },

            "correlations": {

                "columns": num_cols,

                "matrix": corr_matrix

            },

            "metrics": {

                "Linear Regression": {

                    "train": metrics_lr_train,

                    "test": metrics_lr

                },

                "Decision Tree": {

                    "train": metrics_dt_train,

                    "test": metrics_dt

                },

                "Random Forest": {

                    "train": metrics_rf_train,

                    "test": metrics_rf

                }

            },

            "feature_importances": top_importances,

            "coefficients": top_coefs,

            "scatter_plots": scatter_data

        }

        

                                     

        print(f"\n--- Métricas {subject} (Teste) ---")

        for model, met in all_results[subject]["metrics"].items():

            print(f"{model}:")

            print(f"  MAE: {met['test']['MAE']:.3f} | MSE: {met['test']['MSE']:.3f} | RMSE: {met['test']['RMSE']:.3f} | R2: {met['test']['R2']:.3f}")

            

                       

    with open("results_scratch.json", "w", encoding="utf-8") as f:

        json.dump(all_results, f, indent=4, ensure_ascii=False)

        

    print("\nResultados do pipeline salvos com sucesso em 'results_scratch.json'.")



if __name__ == "__main__":

    main()
