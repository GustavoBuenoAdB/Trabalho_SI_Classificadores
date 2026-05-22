import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Gerador de Gráficos para resultados')
parser.add_argument('--arquivo', type=str, help='arquivo csv de entrada')

args = parser.parse_args()
if args.arquivo:
    arquivo = args.arquivo
    df = pd.read_csv(arquivo) 

colunas = df.columns[:4] #escolher quais colunas  #QUANTAS RETAS EU QUERO

for coluna in colunas:
    plt.plot(df[coluna], label=coluna)

#passo do indice x #ESCOLHER O PASSO
#passo = 1
#plt.xticks(np.arange(0, len(df) + 3, passo))

plt.xlabel('Numero da Entradas Treino') #NOMEAR A SAIDA
plt.ylabel('Média Erro Quadratico') #NOMEAR A SAIDA
# Tempo x Número de Árvores (sem Adaboost)
# Tempo x Número de Árvores 
# T
plt.title('Média erro quadratico Aumentando Treino (300)')
plt.legend()
plt.grid(True)
plt.show()