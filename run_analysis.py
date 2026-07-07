import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configurar estilo visual dos gráficos
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

# Paleta de cores premium
primary_color = "#1a2b4c"
secondary_color = "#cda250"
accent_colors = ["#1a2b4c", "#cda250", "#7f8c8d", "#27ae60", "#c0392b"]

def load_data():
    """Carrega os datasets de Matemática e Português."""
    mat_path = os.path.join("data", "student-mat.csv")
    por_path = os.path.join("data", "student-por.csv")
    
    # Os arquivos utilizam ';' como separador
    df_mat = pd.read_csv(mat_path, sep=";")
    df_por = pd.read_csv(por_path, sep=";")
    
    return df_mat, df_por

def perform_eda(df_mat, df_por, output_dir="plots"):
    """Realiza a Análise Exploratória de Dados (EDA) e salva os gráficos."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Distribuição da Variável Nota Final (G3)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    sns.histplot(df_mat['G3'], bins=20, kde=True, color=primary_color, ax=axes[0])
    axes[0].set_title("Distribuição de Notas Finais (G3) - Matemática")
    axes[0].set_xlabel("Nota Final (0 a 20)")
    axes[0].set_ylabel("Frequência")
    
    sns.histplot(df_por['G3'], bins=20, kde=True, color=secondary_color, ax=axes[1])
    axes[1].set_title("Distribuição de Notas Finais (G3) - Português")
    axes[1].set_xlabel("Nota Final (0 a 20)")
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "dist_g3.png"), dpi=300)
    plt.close(fig)
    
    # 2. Avaliar dados ausentes e outliers
    print("--- DADOS AUSENTES ---")
    print("Matemática - Dados ausentes por coluna:")
    print(df_mat.isnull().sum().sum())
    print("Português - Dados ausentes por coluna:")
    print(df_por.isnull().sum().sum())
    
    # 3. Correlações entre variáveis numéricas
    num_cols = df_mat.select_dtypes(include=[np.number]).columns
    
    for name, df in [("Matemática", df_mat), ("Português", df_por)]:
        corr_matrix = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                    cbar_kws={'label': 'Coeficiente de Correlação'}, ax=ax,
                    annot_kws={"size": 8})
        ax.set_title(f"Matriz de Correlação das Variáveis Numéricas - {name}")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"corr_matrix_{name.lower()[:3]}.png"), dpi=300)
        plt.close(fig)

def preprocess_data(df):
    """Realiza o pré-processamento dos dados (One-Hot encoding, normalização e split)."""
    # Separar preditores e alvo
    X = df.drop(columns=['G3'])
    y = df['G3']
    
    # Tratar variáveis categóricas
    # Identificar colunas categóricas e numéricas
    cat_cols = X.select_dtypes(include=['object']).columns
    num_cols = X.select_dtypes(include=[np.number]).columns
    
    # Codificação One-Hot das variáveis categóricas
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    # Dividir dados em treino (70%) e teste (30%)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.3, random_state=42
    )
    
    # Obter colunas codificadas
    encoded_num_cols = [col for col in X_encoded.columns if col in num_cols]
    encoded_cat_cols = [col for col in X_encoded.columns if col not in num_cols]
    
    # Normalizar/Padronizar apenas variáveis numéricas
    scaler = StandardScaler()
    
    # Ajustar e transformar dados de treino e teste nas colunas numéricas
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[encoded_num_cols] = scaler.fit_transform(X_train[encoded_num_cols])
    X_test_scaled[encoded_num_cols] = scaler.transform(X_test[encoded_num_cols])
    
    # Converter para float para evitar problemas de tipo na modelagem
    X_train_scaled = X_train_scaled.astype(float)
    X_test_scaled = X_test_scaled.astype(float)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_encoded.columns

def train_and_evaluate_models(X_train, X_test, y_train, y_test, dataset_name, output_dir="plots"):
    """Treina os modelos e avalia o desempenho."""
    models = {
        "Regressão Linear": LinearRegression(),
        "Árvore de Regressão": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
        "Support Vector Regressor (SVR)": SVR(kernel='rbf', C=10.0, epsilon=0.1)
    }
    
    results = {}
    predictions = {}
    
    for name, model in models.items():
        # Treinar o modelo
        model.fit(X_train, y_train)
        
        # Predições
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        predictions[name] = y_test_pred
        
        # Calcular Métricas
        mae_train = mean_absolute_error(y_train, y_train_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        
        rmse_train = np.sqrt(mse_train)
        rmse_test = np.sqrt(mse_test)
        
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        
        results[name] = {
            "MAE Treino": mae_train, "MAE Teste": mae_test,
            "MSE Treino": mse_train, "MSE Teste": mse_test,
            "RMSE Treino": rmse_train, "RMSE Teste": rmse_test,
            "R² Treino": r2_train, "R² Teste": r2_test,
            "model_obj": model
        }
        
    # Plotar gráficos de comparação para o dataset atual
    plot_evaluation_plots(y_test, predictions, dataset_name, output_dir)
    
    return results

def plot_evaluation_plots(y_true, predictions, dataset_name, output_dir):
    """Gera gráficos de resíduos e real vs previsto para todos os modelos."""
    num_models = len(predictions)
    
    # 1. Gráfico Real vs Previsto
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    for idx, (name, y_pred) in enumerate(predictions.items()):
        ax = axes[idx]
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, color=primary_color, ax=ax)
        # Linha diagonal de referência
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax.set_title(f"{name} - {dataset_name}")
        ax.set_xlabel("Nota Real")
        ax.set_ylabel("Nota Prevista")
        
        # Mostrar métrica R2 no gráfico
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.9, f"R² = {r2:.3f}", transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
        
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"real_vs_previsto_{dataset_name.lower()}.png"), dpi=300)
    plt.close(fig)
    
    # 2. Gráfico de Resíduos
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    for idx, (name, y_pred) in enumerate(predictions.items()):
        ax = axes[idx]
        residuals = y_true - y_pred
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, color=secondary_color, ax=ax)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_title(f"Resíduos: {name} - {dataset_name}")
        ax.set_xlabel("Valores Previstos")
        ax.set_ylabel("Resíduos (Real - Previsto)")
        
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"residuos_{dataset_name.lower()}.png"), dpi=300)
    plt.close(fig)

def get_feature_importances(results, feature_names, dataset_name, output_dir="plots"):
    """Extrai e plota a importância das variáveis do Random Forest e coeficientes da Regressão Linear."""
    # 1. Importância das Variáveis no Random Forest
    rf_model = results["Random Forest Regressor"]["model_obj"]
    importances = rf_model.feature_importances_
    
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Plotar as 15 variáveis mais importantes
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=df_imp.head(15), 
                palette="viridis", ax=ax)
    ax.set_title(f"Top 15 Variáveis Mais Importantes (Random Forest) - {dataset_name}")
    ax.set_xlabel("Grau de Importância")
    ax.set_ylabel("Variável")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"feat_imp_rf_{dataset_name.lower()}.png"), dpi=300)
    plt.close(fig)
    
    # 2. Coeficientes da Regressão Linear
    lr_model = results["Regressão Linear"]["model_obj"]
    coefs = lr_model.coef_
    
    df_coef = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs
    })
    # Filtrar os coeficientes de maior magnitude (positiva ou negativa)
    df_coef['Abs_Coef'] = df_coef['Coefficient'].abs()
    df_coef = df_coef.sort_values(by='Abs_Coef', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=df_coef.head(15), 
                palette="coolwarm", ax=ax)
    ax.set_title(f"Top 15 Coeficientes da Regressão Linear - {dataset_name}")
    ax.set_xlabel("Valor do Coeficiente")
    ax.set_ylabel("Variável")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"coef_lr_{dataset_name.lower()}.png"), dpi=300)
    plt.close(fig)
    
    return df_imp, df_coef

def main():
    print("Iniciando pipeline de análise e modelagem...")
    
    # Carregar dados
    df_mat, df_por = load_data()
    
    # Executar EDA
    print("Realizando Análise Exploratória (EDA)...")
    perform_eda(df_mat, df_por)
    
    # Pré-processamento
    print("Pré-processando dados de Matemática...")
    X_train_mat, X_test_mat, y_train_mat, y_test_mat, feat_names_mat = preprocess_data(df_mat)
    
    print("Pré-processando dados de Português...")
    X_train_por, X_test_por, y_train_por, y_test_por, feat_names_por = preprocess_data(df_por)
    
    # Treinar e avaliar
    print("Treinando e avaliando modelos de Matemática...")
    results_mat = train_and_evaluate_models(X_train_mat, X_test_mat, y_train_mat, y_test_mat, "Matematica")
    
    print("Treinando e avaliando modelos de Português...")
    results_por = train_and_evaluate_models(X_train_por, X_test_por, y_train_por, y_test_por, "Portugues")
    
    # Importância das variáveis
    print("Analisando importância dos atributos...")
    imp_rf_mat, coef_lr_mat = get_feature_importances(results_mat, feat_names_mat, "Matematica")
    imp_rf_por, coef_lr_por = get_feature_importances(results_por, feat_names_por, "Portugues")
    
    # Imprimir tabela resumida de resultados no console
    for name, results in [("MATEMÁTICA", results_mat), ("PORTUGUÊS", results_por)]:
        print(f"\n===== RESULTADOS - {name} =====")
        for model_name, metrics in results.items():
            print(f"\nModelo: {model_name}")
            print(f"  Treino -> MAE: {metrics['MAE Treino']:.3f} | MSE: {metrics['MSE Treino']:.3f} | RMSE: {metrics['RMSE Treino']:.3f} | R²: {metrics['R² Treino']:.3f}")
            print(f"  Teste  -> MAE: {metrics['MAE Teste']:.3f} | MSE: {metrics['MSE Teste']:.3f} | RMSE: {metrics['RMSE Teste']:.3f} | R²: {metrics['R² Teste']:.3f}")
            
    # Salvar resultados em csv para que o gerador de relatórios possa ler
    rows = []
    for subject, res in [("Matematica", results_mat), ("Portugues", results_por)]:
        for model_name, metrics in res.items():
            rows.append({
                "Subject": subject,
                "Model": model_name,
                "MAE_Train": metrics["MAE Treino"],
                "MAE_Test": metrics["MAE Teste"],
                "MSE_Train": metrics["MSE Treino"],
                "MSE_Test": metrics["MSE Teste"],
                "RMSE_Train": metrics["RMSE Treino"],
                "RMSE_Test": metrics["RMSE Teste"],
                "R2_Train": metrics["R² Treino"],
                "R2_Test": metrics["R² Teste"]
            })
    pd.DataFrame(rows).to_csv("results_summary.csv", index=False)
    print("\nResultados salvos em 'results_summary.csv'.")
    print("Pipeline concluído com sucesso!")

if __name__ == "__main__":
    main()
