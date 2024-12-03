# Projeto de Agrupamento de Atividades Humanas com K-means

## Objetivo do Projeto

Este projeto tem como objetivo aplicar o algoritmo K-means para realizar o agrupamento de atividades humanas utilizando dados coletados de sensores de smartphones. O dataset utilizado contém medições de sinais de acelerômetro e giroscópio, permitindo a classificação de diversas atividades físicas. O modelo busca identificar grupos de atividades semelhantes com base em características extraídas desses dados, utilizando técnicas de análise exploratória, normalização, redução de dimensionalidade (PCA) e clustering não supervisionado.

---

## Instruções para Executar o Código

### 1. Pré-requisitos

- **Python 3.x**
- Bibliotecas:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - sklearn

### 2. Como Executar

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   cd seu-repositorio
2. **Crie um ambiente virtual**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Para Windows, use venv\Scripts\activate
3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
4. **Baixe os dados necessários** (dataset "Human Activity Recognition Using Smartphones")<br>
   **e coloque os arquivos no diretório adequado:**
   X_train.txt<br>
   y_train.txt<br>
   features.txt<br>
5. **Execute o código principal:**
   ```bash
   python seu_arquivo.py
