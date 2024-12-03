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
   **e coloque os arquivos no diretório adequado:** <br>
   X_train.txt<br>
   y_train.txt<br>
   features.txt<br><br>
5. **Execute o código principal:**
   ```bash
   python seu_arquivo.py

# Direcionamento para o Relatório Técnico

Para uma análise detalhada de todas as etapas do projeto, incluindo a implementação, avaliação e conclusões, consulte o **Relatório Técnico** que acompanha este repositório. O relatório contém:

- **Introdução** ao problema de reconhecimento de atividades humanas e a justificativa para o uso do algoritmo K-means.
- **Metodologia**, incluindo as etapas de análise exploratória, implementação do K-means e escolha do número de clusters.
- **Resultados** com a avaliação do modelo e análise dos clusters gerados.
- **Discussão** sobre as limitações e possíveis melhorias no modelo, além de sugestões de trabalhos futuros.

O relatório está disponível no arquivo [relatorio.pdf](relatorio.pdf). Clique no arquivo para acessá-lo.

---

# Principais Conclusões e Considerações sobre os Resultados Obtidos

- **Escolha do Número de Clusters**: A escolha do número de clusters foi feita utilizando o método do cotovelo e o **Silhouette Score**, com K=6 se mostrando o valor ideal para separar as atividades de maneira eficaz.
  
- **Qualidade dos Clusters**: O **Silhouette Score** para K=6 indicou uma boa separação e coesão entre os clusters gerados, o que demonstra que o modelo conseguiu identificar padrões distintos nas atividades humanas com boa precisão.

- **Limitações**: O K-means é sensível à inicialização aleatória dos centróides. Embora a inicialização **K-means++** tenha sido utilizada para melhorar a escolha dos centróides, o modelo ainda pode ser sensível a variações nos dados ou na escolha do número de clusters.

- **Trabalhos Futuros**: Em projetos futuros, seria interessante explorar técnicas de clustering mais avançadas, como **DBSCAN** ou **Agglomerative Clustering**, que podem oferecer resultados melhores em situações com dados mais complexos ou com ruídos.
