from itertools import islice

def lerLinha(file, linha_inicio, final):
    for linha in islice(file, linha_inicio, final):
        entrada = linha
    entrada = entrada.split(",")
    del entrada[1]
    del entrada[1]
    return entrada

def lerEntradas(file, linha_inicio, final, n_entradas):
    lista = []
    for i in range(n_entradas):
        lista.append(lerLinha(file, linha_inicio, final))
    return lista

'''
with open('data/02_treino_sinais_vitais_com_label.txt','r',encoding='UTF-8') as f:
    print(lerLinha(f,1,2))
'''