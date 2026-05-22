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

colunas = df.columns[:3] #escolher quais colunas  

for coluna in colunas:
    plt.plot(df[coluna], label=coluna)

plt.xlabel('Índice')
plt.ylabel('Valor')
plt.title('Gráfico X do randomforest')
plt.legend()
plt.grid(True)
plt.show()