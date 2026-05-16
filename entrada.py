from itertools import islice

def lerLinha(file, linha_inicio, final):
    entrada = None
    file.seek(0) #tem que resetar pelo slice

    for linha in islice(file, linha_inicio, final):
        entrada = linha
    entrada = entrada.split(",")
    del entrada[1]
    del entrada[1]
    entrada = [float(x) for x in entrada] #convertendo de string pra float
    entrada[5] = int(entrada[5]) #a label é o unico int
    return entrada

def lerEntradas(file, linha_inicio, n_entradas):
    lista = []
    for i in range(linha_inicio, linha_inicio + n_entradas):
        lista.append(lerLinha(file, i, i+2)) # +2 pq vai ir em um range e tem que ter o de consideração
    return lista

'''
with open('data/02_treino_sinais_vitais_com_label.txt','r',encoding='UTF-8') as f:
    print(lerLinha(f,1,2))
'''