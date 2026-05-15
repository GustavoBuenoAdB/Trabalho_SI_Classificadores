from itertools import islice

def lerLinha(arquivo, linha_inicio):
    with open(arquivo, "r", encoding="utf-8") as f:
        for linha in islice(f, linha_inicio, linha_inicio + 1):
            entrada = linha
    return entrada


