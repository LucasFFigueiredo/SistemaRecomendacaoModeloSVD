import pandas as pd
import numpy as np
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import accuracy

try:
    df = pd.read_csv('online_retail.csv', encoding='latin1')
except FileNotFoundError:
    print("Erro: Arquivo 'online_retail.csv' não encontrado. Por favor, carregue o arquivo no seu ambiente.")

except Exception as e:
    print(f"Ocorreu um erro ao carregar o arquivo: {e}")
    

total_nulos = df['CustomerID'].isnull().sum()
print(f"\nTotal de linhas com 'CustomerID' nulo: {total_nulos}")
print(f"Porcentagem de nulos: {(total_nulos / len(df) * 100):.2f}%")

df.dropna(subset=['CustomerID'], inplace=True)

canceladas = df[df['InvoiceNo'].astype(str).str.startswith('C')]

df_limpo = df[~df['InvoiceNo'].astype(str).str.startswith('C')].copy()

df_limpo = df_limpo[df_limpo['Quantity'] > 0]
df_limpo = df_limpo[df_limpo['UnitPrice'] > 0]
print(f"\nTotal de linhas após garantir quantidades e preços positivos: {len(df_limpo)}")


df_limpo['TotalValue'] = df_limpo['Quantity'] * df_limpo['UnitPrice']
df_limpo['Rating_Log'] = np.log1p(df_limpo['TotalValue'])


df_limpo['CustomerID'] = df_limpo['CustomerID'].astype(int)

print("\nDataFrame pronto para modelagem (coluna 'TotalValue' como Rating):")
print(df_limpo[['CustomerID', 'StockCode', 'TotalValue']].head())


interaction_matrix = df_limpo.groupby(['CustomerID', 'StockCode'])['Rating_Log'].sum().reset_index()


user_item_matrix = interaction_matrix.pivot_table(
    index='CustomerID', 
    columns='StockCode', 
    values='Rating_Log'
).fillna(0)

print("\nFormato da Matriz Usuário-Item (Base para o Modelo):")
print(user_item_matrix.head())
print(f"\nDimensões da Matriz: {user_item_matrix.shape} (Clientes por Produtos)")

min_rating = df_limpo['Rating_Log'].min()
max_rating = df_limpo['Rating_Log'].max()

reader = Reader(rating_scale=(min_rating, max_rating))

data = Dataset.load_from_df(
    df_limpo[['CustomerID', 'StockCode', 'Rating_Log']], 
    reader
)

trainset, testset = train_test_split(data, test_size=0.20, random_state=42)

print(f"Número de observações (interações) no conjunto de treino: {trainset.n_ratings}")
print(f"Número de observações (interações) no conjunto de teste: {len(testset)}")

algo = SVD(random_state=42)

print("\nIniciando o treinamento do modelo SVD...")
algo.fit(trainset)
print("Treinamento concluído.")

predictions = algo.test(testset)

rmse = accuracy.rmse(predictions, verbose=True)

print(f"\nO modelo alcançou um RMSE de: {rmse:.4f}")

cliente_id = df_limpo['CustomerID'].iloc[0] 

itens_comprados = df_limpo[df_limpo['CustomerID'] == cliente_id]['StockCode'].unique()
todos_itens = df_limpo['StockCode'].unique()
itens_a_recomendar = [item for item in todos_itens if item not in itens_comprados]


previsoes = []
for item in itens_a_recomendar:

    previsao_rating = algo.predict(uid=cliente_id, iid=item).est
    previsoes.append((item, previsao_rating))

top_10_recomendacoes = sorted(previsoes, key=lambda x: x[1], reverse=True)[:10]

print(f"\n--- Top 10 Recomendações para o Cliente ID: {cliente_id} ---")
for i, (item_code, predicted_rating) in enumerate(top_10_recomendacoes):
    print(f"{i+1}. Item (StockCode): {item_code} | Rating Previsto (TotalValue): {predicted_rating:.2f}")