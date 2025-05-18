import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

df = pd.read_csv("countries of the world.csv")
df.columns = df.columns.str.strip()
df = df.replace(',', '', regex=True)

df = df.rename(columns={
    'Country': 'Country',
    'GDP ($ per capita)': 'GDP_per_capita',
    'Area (sq. mi.)': 'Area',
    'Literacy (%)': 'Literacy',
    'Net migration': 'Net_migration'
})

features = ['Population', 'Area', 'GDP_per_capita', 'Literacy',
            'Birthrate', 'Deathrate', 'Net_migration', 'Climate',
            'Agriculture', 'Industry', 'Service']

for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df[features] = df[features].fillna(df[features].mean())

scaler = StandardScaler()
data_scaled = scaler.fit_transform(df[features])

def get_similar_countries(country_name, top_n=5):
    idx = df[df['Country'].str.strip().str.lower() == country_name.lower()].index
    if not idx.empty:
        query_vec = data_scaled[idx[0]].reshape(1, -1)
        similarities = cosine_similarity(query_vec, data_scaled)[0]
        sim_df = pd.DataFrame({'Country': df['Country'], 'Similarity': similarities})
        sim_df = sim_df[df['Country'].str.strip().str.lower() != country_name.lower()]
        
        return sim_df.sort_values(by='Similarity', ascending=False).head(top_n)
    else:
        print(f"Country '{country_name}' not found.")
        return pd.DataFrame()


print("\nTop Similar Countries to Kazakhstan:")
print(get_similar_countries("Kazakhstan"))

print("\nTop Similar Countries to Canada:")
print(get_similar_countries("Canada"))

print("\nTop Similar Countries to Morocco:")
print(get_similar_countries("Morocco"))

def plot_similar_countries(query_country, top_n=5):
    sim_df = get_similar_countries(query_country, top_n)
    sim_df = sim_df[sim_df['Country'].str.lower() != query_country.lower()]
    plt.figure(figsize=(8, 5))
    sns.barplot(data=sim_df, x='Similarity', y='Country', palette='viridis')
    plt.title(f"Top {top_n} Countries Most Similar to {query_country}")
    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Country")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()

plot_similar_countries("Kazakhstan", top_n=5)
plot_similar_countries("Canada", top_n=5)
plot_similar_countries("Morocco", top_n=5)

silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    preds = km.fit_predict(data_scaled)
    score = silhouette_score(data_scaled, preds)
    silhouette_scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(k_range, silhouette_scores, marker='o')
plt.title("Silhouette Scores for Different Values of k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(data_scaled)

print("\nCluster Feature Averages:\n")
print(df.groupby('Cluster')[features].mean().round(2))

print("\nNumber of countries in each cluster:")
print(df['Cluster'].value_counts().sort_index())

pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_scaled)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', s=100, edgecolor='k', alpha=0.8)

for i in range(len(df)):
    plt.text(df['PCA1'][i], df['PCA2'][i], df['Country'].iloc[i], fontsize=6, alpha=0.6)

plt.title(f"K-Means Clustering of Countries (k={optimal_k}) - PCA Projection")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title='Cluster', loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"\nFinal Silhouette Score (k={optimal_k}): {silhouette_score(data_scaled, df['Cluster']):.2f}")

cluster_means = df.groupby('Cluster')[features].mean().round(2)

unique_clusters = df['Cluster'].unique()
palette = sns.color_palette('tab10', len(unique_clusters))

for cluster_id in sorted(unique_clusters):
    cluster_data = df[df['Cluster'] == cluster_id]
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=cluster_data, x='PCA1', y='PCA2', color=palette[cluster_id], s=100, edgecolor='k')
    
    for i in range(len(cluster_data)):
        plt.text(cluster_data['PCA1'].iloc[i], cluster_data['PCA2'].iloc[i], cluster_data['Country'].iloc[i],
                 fontsize=6, alpha=0.7)
    
    plt.title(f"PCA Plot – Cluster {cluster_id} Only")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

