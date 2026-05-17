
import random
from scipy.special import expit #exipit é a sigmoid

#numeros de entrada e saidas de cada camada
N_ENT_C1 = 3
N_NEU_C1 = 4

N_ENT_C2 = 4
N_NEU_C2 = 6

N_ENT_C3 = 6
N_NEU_C3 = 2

#flags de "debug" = printf
DEBUG_CAM = False
DEBUG_NEU = True

def f_ativacao(x):
	return expit(x)

class Neuronio:
	def __init__(self, n_entradas):
		self.n_entradas = n_entradas
		self.pesos = []
		self.vies = random.random()
		# inicializa os pesos aleatoriamente
		for i in range (n_entradas + 1):
			self.pesos.append(random.random())


	def processa(self, entradas):
		som = 0
		n = len(entradas)
		for i in range(1, n + 1):
			som += entradas[i-1] * self.pesos[i]		
		som -= self.pesos[0] # esse aqui é o vies (indice 0 do vetor de pesos)

		if (DEBUG_NEU):
			print(f" - entrada: {entradas} \n - pesos: {self.pesos} \n - soma: {som}")

		return f_ativacao(som)

class Camada:
	def __init__(self, n_entradas, n_saidas):
		self.n_entradas = n_entradas
		self.n_saidas = n_saidas #que tbm é o numero de neuronios
		self.neuronios = []

		# inicializa os neuronios da camada
		for i in range(n_saidas):
			self.neuronios.append(Neuronio(self.n_entradas))

	def processa(self, entrada):
		# processa a entrada em cada neuronio e devolve o vetor de saida
		saida = []
		for i in range(self.n_saidas):
			if (DEBUG_NEU):
				print(f"neuronio {i}")
			saida.append((self.neuronios[i]).processa(entrada)) #acessa neuronio, processa e add na lista
			
			if (DEBUG_NEU):
				print(f" - ativacao {saida[i]}")
		
		if (DEBUG_CAM):
			print(f" - entrada: {entrada} \n - saida: {saida}")

		return saida

class MLP:
	def __init__(self):
		self.camadas = []

	def add_camada(self, camada):
		self.camadas.append(camada)

	def processa(self, entrada):
		saida = entrada
		for i in range(len(self.camadas)):
			if (DEBUG_CAM):
				print(f"camada {i}")
			saida = (self.camadas[i].processa(saida)) # enquanto tiver camada a saida de um é a entrada dotro

		return saida

def main():
	mlp = MLP()

	mlp.add_camada(Camada(N_ENT_C1, N_NEU_C1))
	mlp.add_camada(Camada(N_ENT_C2, N_NEU_C2))
	mlp.add_camada(Camada(N_ENT_C3, N_NEU_C3))

	print(mlp.processa((-2, 1, 3)))

if __name__ == "__main__":
	main()


# TODO: aqui ta implementada só a base da base, não treina nem pega de arquivo, ainda é bem xunxa, n sei inclusive se o vies o - la ta sendo calculado certinho
# pra ser sincero meio que esqueci dele, tem que ver isso, adicionar leitura dos dados certinhos la, e programar a tal da BACKPROPAGATION, mas aqui é só 
# o esquletinho pra brincar de ajustar parametro de camada