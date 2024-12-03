import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer

# 1. Carregando os dados
X_train = pd.read_csv('X_train.txt', sep='\s+', header=None)
y_train = pd.read_csv('y_train.txt', sep='\s+', header=None)
features = pd.read_csv('features.txt', sep='\s+', header=None)

# Atribuindo os nomes das colunas
X_train.columns = features[1].values

# 2. Verificando se h� colunas n�o num�ricas e tratando os valores ausentes
print(X_train.describe())
print(X_train.isnull().sum())  # Verifica a quantidade de valores ausentes

# Verificando se h� colunas n�o num�ricas
non_numeric_columns = X_train.select_dtypes(exclude=[np.number]).columns
print(f"Colunas n�o num�ricas: {non_numeric_columns}")

# Caso haja colunas n�o num�ricas, podemos decidir se remov�-las ou convert�-las (ex: para codifica��o)
# Neste caso, vamos remov�-las, mas voc� pode escolher outro tratamento dependendo do seu caso.
X_train_cleaned = X_train.drop(columns=non_numeric_columns)

# Imputando valores ausentes com a m�dia, apenas nas colunas num�ricas
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train_cleaned)  # Imputando valores ausentes

# Convertendo de volta para DataFrame
X_train_imputed_df = pd.DataFrame(X_train_imputed, columns=X_train_cleaned.columns)

# 3. Visualiza��o das distribui��es das primeiras vari�veis
X_train_imputed_df.iloc[:, :5].hist(figsize=(12, 10))
plt.show()

# Matriz de correla��o
corr_matrix = X_train_imputed_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt='.2f')
plt.show()

# 4. Normaliza��o dos dados antes de aplicar o PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed_df)

# 5. Redu��o de Dimensionalidade com PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

# Visualizando os dados ap�s PCA
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], s=5)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Dados ap�s PCA')
plt.show()

# 6. Escolha do N�mero de Clusters (M�todo do Cotovelo e Silhouette Score)
inertia = []
sil_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_train_scaled)
    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_train_scaled, kmeans.labels_))

# M�todo do Cotovelo
plt.plot(K_range, inertia, marker='o')
plt.title('M�todo do Cotovelo')
plt.xlabel('N�mero de Clusters (K)')
plt.ylabel('In�rcia')
plt.show()

# Silhouette Score
plt.plot(K_range, sil_scores, marker='o')
plt.title('Silhouette Score para diferentes K')
plt.xlabel('N�mero de Clusters (K)')
plt.ylabel('Silhouette Score')
plt.show()

# 7. Implementando o K-means com K=6 (ou o valor ideal encontrado)
k = 6
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
kmeans.fit(X_train_scaled)
clusters = kmeans.labels_  # R�tulos dos clusters (valores inteiros de 0 a k-1)

# Verificando os valores dos clusters para garantir que s�o inteiros
print("Valores dos clusters (r�tulos):")
print(np.unique(clusters))  # Isso deve imprimir uma sequ�ncia de n�meros de 0 a k-1, por exemplo, [0, 1, 2, 3, 4, 5]

# Visualizando os clusters com PCA
X_train_pca = pca.fit_transform(X_train_scaled)

# Agora usamos 'clusters' para colorir os pontos. Como 'clusters' cont�m r�tulos inteiros de 0 a k-1, ele pode ser usado diretamente como argumento 'c' para colorir os pontos.
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=clusters, cmap='viridis', s=5)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title(f'Clusters K-means com K={k}')
plt.colorbar(label='Cluster ID')  # Adiciona uma barra de cores para mostrar os r�tulos dos clusters
plt.show()

# 8. Avalia��o do Modelo (Silhouette Score)
sil_score = silhouette_score(X_train_scaled, kmeans.labels_)
print(f'Silhouette Score: {sil_score}')

# 9. Analisando os Centros dos Clusters
print("Centros dos Clusters:")
print(kmeans.cluster_centers_)

# 10. Visualiza��o com t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_train_tsne = tsne.fit_transform(X_train_scaled)

plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=clusters, cmap='viridis', s=5)
plt.title(f'Clusters K-means com K={k} (t-SNE)')
plt.show()

# 11. Repetindo o K-means para verificar a estabilidade
kmeans_multiple_runs = []
for _ in range(10):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=np.random.randint(0, 10000))
    kmeans.fit(X_train_scaled)
    kmeans_multiple_runs.append(kmeans.labels_)

# Verificando a consist�ncia dos r�tulos entre as execu��es
print(kmeans_multiple_runs[0] == kmeans_multiple_runs[1])  # Verifique a consist�ncia entre as execu��es