# Predição de Notas de Estudantes - Inteligência Artificial II

Este repositório contém o trabalho acadêmico de **Inteligência Artificial II** desenvolvido para o curso de **Ciência da Computação** da **Unifucamp**. 

O objetivo do projeto é prever a nota final dos estudantes (atributo `G3`) com base em características demográficas, sociais e acadêmicas utilizando a base de dados real [Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance) da UCI (coletado em duas escolas secundárias de Portugal para as matérias de **Matemática** e **Língua Portuguesa**).

---

## 🛠️ Arquitetura do Projeto (Modelagem do Zero)

Para garantir portabilidade e demonstrar os fundamentos matemáticos dos algoritmos sem depender de wrappers externos prontas (e evitar bloqueios de DLLs dinâmicas em ambientes restritos), todas as etapas de modelagem foram implementadas **do zero (scratch)** em Python puro.

### Estrutura de Arquivos
* 📁 `data/`: Diretório contendo os arquivos originais baixados (`student-mat.csv`, `student-por.csv` e documentações).
* 📄 `download_data.py`: Script para baixar e descompactar automaticamente o dataset diretamente da UCI.
* 📄 `ml_from_scratch.py`: Biblioteca desenvolvida em Python puro contendo:
  * O pré-processador (`StandardScalerScratch` e `OneHotEncoderScratch`).
  * A implementação dos modelos (`LinearRegressionGD`, `DecisionTreeRegressorScratch` e `RandomForestRegressorScratch`).
  * As métricas de avaliação (`MAE`, `MSE`, `RMSE` e `R²`).
* 📄 `run_analysis_scratch.py`: Pipeline principal que processa os dados, executa o treinamento dos modelos e exporta as coordenadas de plotagem e estatísticas (`results_scratch.json`).
* 📄 `generate_report.py`: Gerador automático de relatório que desenha **gráficos vetoriais nativos** (sem o uso de matplotlib) e gera o arquivo PDF final.
* 📄 `student_analysis.ipynb`: Notebook Jupyter interativo pronto para ser executado (localmente ou no Google Colab) contendo plotagens com `matplotlib` e `seaborn`.
* 📄 `relatorio_predicao_notas.pdf`: O produto final contendo os gráficos e análises visuais interpretadas.

---

## 🚀 Como Rodar o Projeto

### 1. Requisitos Prévios
Certifique-se de ter o Python 3 instalado em sua máquina.

### 2. Instalação das Dependências

Para executar apenas os scripts principais e gerar o PDF do relatório (usando as classes do zero), você precisa apenas de:
```bash
pip install reportlab requests
```

Caso queira executar também o Notebook Jupyter interativo com as plotagens padrão, instale o restante dos pacotes científicos:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn notebook
```

### 3. Passo a Passo de Execução

Abra o seu terminal na pasta do projeto e execute os scripts na ordem abaixo:

1. **Baixar e organizar a base de dados:**
   ```bash
   python download_data.py
   ```
2. **Treinar os modelos e salvar os resultados da análise:**
   ```bash
   python run_analysis_scratch.py
   ```
3. **Gerar o arquivo PDF contendo os gráficos e interpretações:**
   ```bash
   python generate_report.py
   ```

Ao término, o arquivo `relatorio_predicao_notas.pdf` será gerado/atualizado na raiz do projeto.

---

## 📈 Resultados Comparativos Obtidos (Teste)

| Disciplina | Modelo | MAE | MSE | RMSE | R² |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Matemática** | Regressão Linear | 1.442 | 4.014 | 2.003 | 0.737 |
| **Matemática** | Árvore de Decisão | 1.191 | 3.136 | 1.771 | **0.794** |
| **Matemática** | Random Forest | 1.412 | 3.903 | 1.976 | 0.744 |
| **Português** | Regressão Linear | 0.795 | 1.749 | 1.322 | **0.859** |
| **Português** | Árvore de Decisão | 0.857 | 2.298 | 1.516 | 0.815 |
| **Português** | Random Forest | 0.955 | 2.441 | 1.562 | 0.804 |

---

## 👥 Autores
* **Lucas Firmino Rodrigues**
* **Carlos Eduardo Mendes**

*Unifucamp - Ciência da Computação (2026)*
