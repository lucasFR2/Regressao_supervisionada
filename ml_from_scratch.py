import csv

import random

import math



                                                  

random.seed(42)



def train_test_split(X, y, test_size=0.3, random_state=42):

    """Divide os dados em conjuntos de treino e teste."""

    random.seed(random_state)

    n = len(X)

    indices = list(range(n))

    random.shuffle(indices)

    

    test_count = int(n * test_size)

    test_indices = indices[:test_count]

    train_indices = indices[test_count:]

    

    X_train = [X[i] for i in train_indices]

    X_test = [X[i] for i in test_indices]

    y_train = [y[i] for i in train_indices]

    y_test = [y[i] for i in test_indices]

    

    return X_train, X_test, y_train, y_test



class StandardScalerScratch:

    """Padronizador numérico (Z-score) implementado do zero."""

    def __init__(self):

        self.means = []

        self.stds = []



    def fit(self, X):

                                                             

        n_features = len(X[0])

        n_samples = len(X)

        self.means = []

        self.stds = []

        

        for j in range(n_features):

            col = [X[i][j] for i in range(n_samples)]

            mean = sum(col) / n_samples

                                                                                    

            variance = sum((x - mean) ** 2 for x in col) / n_samples

            std = math.sqrt(variance) if variance > 0 else 1.0

            self.means.append(mean)

            self.stds.append(std)

            

    def transform(self, X):

        X_trans = []

        for row in X:

            new_row = []

            for j in range(len(row)):

                new_row.append((row[j] - self.means[j]) / self.stds[j])

            X_trans.append(new_row)

        return X_trans



    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)



class OneHotEncoderScratch:

    """Codificador categórico One-Hot implementado do zero."""

    def __init__(self):

        self.categories_map = {}                                                                       

        self.cat_cols = []

        self.num_cols = []

        self.feature_names = []



    def fit(self, df_data, cat_cols, num_cols):

        self.cat_cols = cat_cols

        self.num_cols = num_cols

        self.categories_map = {}

        

                                                                  

        for col in cat_cols:

            unique_vals = sorted(list(set(row[col] for row in df_data)))

                                                                                        

            if len(unique_vals) > 1:

                self.categories_map[col] = unique_vals[1:]

            else:

                self.categories_map[col] = unique_vals

                

                                              

        self.feature_names = list(num_cols)

        for col in cat_cols:

            for val in self.categories_map[col]:

                self.feature_names.append(f"{col}_{val}")



    def transform(self, df_data):

        X = []

        for row in df_data:

            new_row = []

                                          

            for col in self.num_cols:

                new_row.append(float(row[col]))

                                                    

            for col in self.cat_cols:

                val = row[col]

                for cat in self.categories_map[col]:

                    new_row.append(1.0 if val == cat else 0.0)

            X.append(new_row)

        return X



def load_and_preprocess_scratch(file_path):

    """Carrega o arquivo CSV e realiza o pré-processamento manual."""

                  

    data = []

    with open(file_path, mode='r', encoding='utf-8') as f:

        reader = csv.DictReader(f, delimiter=';')

        for row in reader:

            data.append(row)

            

                                  

    all_cols = list(data[0].keys())

                 

    target_col = 'G3'

    features_cols = [c for c in all_cols if c != target_col]

    

                                     

                                                     

    num_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 

                'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']

    cat_cols = [c for c in features_cols if c not in num_cols]

    

                 

    encoder = OneHotEncoderScratch()

    encoder.fit(data, cat_cols, num_cols)

    X = encoder.transform(data)

    y = [float(row[target_col]) for row in data]

    

    return X, y, encoder.feature_names, num_cols, cat_cols



                                                   



class LinearRegressionGD:

    """Regressão Linear via Gradiente Descendente."""

    def __init__(self, lr=0.01, epochs=1000):

        self.lr = lr

        self.epochs = epochs

        self.weights = []

        self.bias = 0.0



    def fit(self, X, y):

        n_samples = len(X)

        n_features = len(X[0])

        

                                            

        self.weights = [0.0] * n_features

        self.bias = 0.0

        

        for epoch in range(self.epochs):

            dw = [0.0] * n_features

            db = 0.0

            

            for i in range(n_samples):

                xi = X[i]

                yi = y[i]

                

                          

                y_pred = sum(xi[j] * self.weights[j] for j in range(n_features)) + self.bias

                error = y_pred - yi

                

                                     

                for j in range(n_features):

                    dw[j] += error * xi[j]

                db += error

                

                                  

            for j in range(n_features):

                self.weights[j] -= (self.lr * 2.0 / n_samples) * dw[j]

            self.bias -= (self.lr * 2.0 / n_samples) * db

            

    def predict(self, X):

        predictions = []

        n_features = len(self.weights)

        for row in X:

            y_pred = sum(row[j] * self.weights[j] for j in range(n_features)) + self.bias

            predictions.append(y_pred)

        return predictions



class Node:

    """Nó para a Árvore de Decisão de Regressão."""

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):

        self.feature = feature                                              

        self.threshold = threshold                      

        self.left = left                                 

        self.right = right                              

        self.value = value                               



    def is_leaf(self):

        return self.value is not None



class DecisionTreeRegressorScratch:

    """Árvore de Regressão construída do zero."""

    def __init__(self, max_depth=5, min_samples_split=5, max_features=None):

        self.max_depth = max_depth

        self.min_samples_split = min_samples_split

        self.max_features = max_features                                                

        self.root = None



    def fit(self, X, y):

        self.root = self._build_tree(X, y, depth=0)



    def _calculate_sse(self, y):

        """Calcula a soma dos erros quadráticos (Soma de Quadrados dos Resíduos)."""

        if not y:

            return 0.0

        mean = sum(y) / len(y)

        return sum((val - mean) ** 2 for val in y)



    def _best_split(self, X, y, feature_indices):

        """Encontra a melhor divisão para o nó atual."""

        best_sse = float('inf')

        best_idx = None

        best_thresh = None

        

        n_samples = len(X)

        if n_samples < self.min_samples_split:

            return None, None

            

        for idx in feature_indices:

                                                         

            values = [row[idx] for row in X]

            unique_vals = sorted(list(set(values)))

            

                                                             

            if len(unique_vals) <= 1:

                continue

                

                                                                            

                                                                                                        

            thresholds = []

            if len(unique_vals) > 10:

                                                             

                step = len(unique_vals) // 10

                for k in range(1, 10):

                    thresholds.append(unique_vals[k * step])

            else:

                for k in range(len(unique_vals) - 1):

                    thresholds.append((unique_vals[k] + unique_vals[k+1]) / 2.0)

                    

            for thresh in thresholds:

                                  

                left_y = [y[i] for i in range(n_samples) if X[i][idx] <= thresh]

                right_y = [y[i] for i in range(n_samples) if X[i][idx] > thresh]

                

                if len(left_y) == 0 or len(right_y) == 0:

                    continue

                    

                sse = self._calculate_sse(left_y) + self._calculate_sse(right_y)

                

                if sse < best_sse:

                    best_sse = sse

                    best_idx = idx

                    best_thresh = thresh

                    

        return best_idx, best_thresh



    def _build_tree(self, X, y, depth=0):

        n_samples = len(X)

        n_features = len(X[0])

        

                             

        if (depth >= self.max_depth or 

            n_samples < self.min_samples_split or 

            len(set(y)) == 1):

            leaf_value = sum(y) / len(y) if y else 0.0

            return Node(value=leaf_value)

            

                                                                                                     

        if self.max_features is not None:

            max_feats = min(n_features, self.max_features)

            feature_indices = random.sample(range(n_features), max_feats)

        else:

            feature_indices = list(range(n_features))

            

                              

        best_idx, best_thresh = self._best_split(X, y, feature_indices)

        

                                                              

        if best_idx is None:

            leaf_value = sum(y) / len(y) if y else 0.0

            return Node(value=leaf_value)

            

                                

        left_X = [X[i] for i in range(n_samples) if X[i][best_idx] <= best_thresh]

        left_y = [y[i] for i in range(n_samples) if X[i][best_idx] <= best_thresh]

        

        right_X = [X[i] for i in range(n_samples) if X[i][best_idx] > best_thresh]

        right_y = [y[i] for i in range(n_samples) if X[i][best_idx] > best_thresh]

        

        left_child = self._build_tree(left_X, left_y, depth + 1)

        right_child = self._build_tree(right_X, right_y, depth + 1)

        

        return Node(feature=best_idx, threshold=best_thresh, left=left_child, right=right_child)



    def predict(self, X):

        return [self._predict_row(self.root, row) for row in X]



    def _predict_row(self, node, row):

        if node.is_leaf():

            return node.value

        if row[node.feature] <= node.threshold:

            return self._predict_row(node.left, row)

        return self._predict_row(node.right, row)



class RandomForestRegressorScratch:

    """Random Forest Regressor construído do zero."""

    def __init__(self, n_estimators=50, max_depth=6, min_samples_split=5, max_features=None, random_state=42):

        self.n_estimators = n_estimators

        self.max_depth = max_depth

        self.min_samples_split = min_samples_split

        self.max_features = max_features

        self.random_state = random_state

        self.trees = []



    def fit(self, X, y):

        random.seed(self.random_state)

        n_samples = len(X)

        n_features = len(X[0])

        

                                                                                                   

        if self.max_features is None:

            self.max_features = max(1, n_features // 3)

            

        self.trees = []

        for i in range(self.n_estimators):

                                               

            bootstrap_indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]

            bootstrap_X = [X[idx] for idx in bootstrap_indices]

            bootstrap_y = [y[idx] for idx in bootstrap_indices]

            

                                      

            tree = DecisionTreeRegressorScratch(

                max_depth=self.max_depth,

                min_samples_split=self.min_samples_split,

                max_features=self.max_features

            )

            tree.fit(bootstrap_X, bootstrap_y)

            self.trees.append(tree)



    def predict(self, X):

                                                                 

        predictions = []

        n_samples = len(X)

        

                                                                                 

        tree_preds = [tree.predict(X) for tree in self.trees]

        

                                               

        for i in range(n_samples):

            sample_preds = [tree_preds[t][i] for t in range(self.n_estimators)]

            predictions.append(sum(sample_preds) / self.n_estimators)

            

        return predictions



    def get_feature_importances(self, n_features):

        """Calcula a importância das variáveis baseada na frequência de uso nos nós internos das árvores."""

        importance = [0.0] * n_features

        for tree in self.trees:

            self._accumulate_importance(tree.root, importance)

            

                                 

        total = sum(importance)

        if total > 0:

            importance = [val / total for val in importance]

        return importance



    def _accumulate_importance(self, node, importance):

        if node is None or node.is_leaf():

            return

                                                                   

        importance[node.feature] += 1.0

        self._accumulate_importance(node.left, importance)

        self._accumulate_importance(node.right, importance)



                                                    



def calculate_metrics(y_true, y_pred):

    """Calcula as métricas MAE, MSE, RMSE e R²."""

    n = len(y_true)

    

    mae = sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n

    mse = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) / n

    rmse = math.sqrt(mse)

    

    mean_y = sum(y_true) / n

    ss_tot = sum((y_true[i] - mean_y) ** 2 for i in range(n))

    ss_res = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n))

    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    

    return {

        "MAE": mae,

        "MSE": mse,

        "RMSE": rmse,

        "R2": r2

    }
