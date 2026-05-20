import entrada as ent
import random
import numpy
from scipy.special import expit #exipit é a sigmoid

N_ENTRADAS = 1499 # [1, 1500] = 1499
N_ENTRADAS_TREINO = 1000
N_EPOCAS = 5

TAM_MINI = 10 #queisso???
TAXA_APRENDIZAGEM = 0.01

MIN_PESO = -5.0
MAX_PESO = 5.0

#numeros de entrada e saidas de cada camada
N_ENT_C1 = 3
N_NEU_C1 = 6

N_ENT_C2 = 6
N_NEU_C2 = 5

N_ENT_C3 = 5
N_NEU_C3 = 4

#flags de "debug" = print
DEBUG_CAM = False
DEBUG_NEU = False
DEBUG_TRN = True

#normalizações
MIN_QPA = -10
MAX_QPA = 10
MIN_PUL = 0
MAX_PUL = 200
MIN_RSP = 0
MAX_RSP = 22


saidas_esperadas = [[1.00, 0.00, 0.00, 0.00], [0.00, 1.00, 0.00, 0.00], [0.00, 0.00, 1.00, 0.00], [0.00, 0.00, 0.00, 1.00]] #alvos pra treino do back

def f_ativacao(x):
	return expit(x)

def derivada_sigmoid(x):
	s = f_ativacao(x)
	return s * (1 - s)

class Neuronio:
	def __init__(self, n_entradas, taxa_aprendizado):
		self.n_entradas = n_entradas
		self.pesos = []

		self.ult_ent = []
		self.ult_z = [] #essa é a ultima soma de pesos vezes entradas, antes da sigmoid
		self.ult_a = []
		self.delta = 0 # derivada de Custo em relação a derivada de ativação da camada.
		self.taxa_apr = taxa_aprendizado

		#self.ultima_ativacao = [] #essa é a ultima saida?
		#self.ultima_som = [] #antes da sigmoid?
		
		# inicializa os pesos aleatoriamente
		for i in range (n_entradas + 1):
			self.pesos.append(random.uniform(0.1, 0.9))

	def processa(self, entradas):
		
		som = 0
		n = len(entradas)
		for i in range(1, n + 1):
			som += entradas[i-1] * self.pesos[i]		
		som -= self.pesos[0] # esse aqui é o vies (indice 0 do vetor de pesos)
		ativacao = f_ativacao(som)

		#armazenando os valores do processamento
		self.ult_ent = entradas
		self.ult_z = som
		self.ult_a = ativacao

		if (DEBUG_NEU):
			print(f" - entrada: {entradas} \n - pesos: {self.pesos} \n - soma: {som}")
		return ativacao
	
	def atualiza_pesos(self, atualizacao): #OBS: taxa pode virar parametro interno se nois for adicionar MOMENTUM
		if (DEBUG_TRN):
			print(f"- pesos......: {self.pesos}")
			print(f"- atualização: {atualizacao}")

		for i in range(self.n_entradas):
			self.pesos[i] = self.pesos[i] - atualizacao[i]
			
			#normalizando os pesos
			if (self.pesos[i] < MIN_PESO):
				self.pesos[i] = MIN_PESO
			elif(self.pesos[i] > MAX_PESO):
				self.pesos[i] = MAX_PESO

		if (DEBUG_TRN):
			print(f"- pesos_atualizados: {self.pesos}\n")
	
class Camada:
	def __init__(self, n_entradas, n_saidas):
		
		self.n_entradas = n_entradas
		self.n_saidas = n_saidas #que tbm é o numero de neuronios
		self.neuronios = []
		self.deltas = [] # derivada de Custo em relação a derivada de ativação da camada.

		# inicializa os neuronios da camada
		for i in range(n_saidas):
			self.neuronios.append(Neuronio(self.n_entradas, TAXA_APRENDIZAGEM))
			self.deltas.append(0) 

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
	
	def atualiza_pesos(self, prox_camada):
		i = 0
		for n in self.neuronios:
			atualizacao = []

			# attualização do vies do neuronio n
			erro_propagado_vies = 0
			for n_prox in prox_camada.neuronios:
				c = (derivada_sigmoid(n_prox.ult_z) * n_prox.delta) 
				erro_propagado_vies += c

			atualizacao.append(n.taxa_apr * erro_propagado_vies) # [0] do vetor de pesos

			# atualização de cada peso do neuronio n
			erro_propagado = 0 #erro propagado por n para a proxima camada
			for n_prox in prox_camada.neuronios:
				c = (n.ult_a) * (derivada_sigmoid(n_prox.ult_z)) * n_prox.delta 
				erro_propagado += c 

			for j in range(len(n.pesos) - 1):
				atualizacao.append(erro_propagado * n.ult_ent[j] * n.taxa_apr) #atualização ta sempre positiva, pq?

			n.delta = derivada_sigmoid(n.ult_z) * erro_propagado

			n.atualiza_pesos(atualizacao)

			i += 1

	def calcula_deltas_saida(self, saida_esperada):
		i = 0
		for n in self.neuronios:
			n.delta = 2*(n.ult_a - saida_esperada[i])
			i += 1
	
class MLP:
	def __init__(self):
		self.camadas = []

	def add_camada(self, camada):
		self.camadas.append(camada)

	def processa(self, entrada):
		saida = None
		for i in range(len(self.camadas)):
			if (DEBUG_CAM):
				print(f"camada {i}")
			saida = (self.camadas[i].processa(entrada)) # enquanto tiver camada a saida de um é a entrada dotro
			entrada = saida
		return saida
	
	def backpropagation(self, saida_esperada):
		n_cam = len(self.camadas)

		self.camadas[n_cam - 1].calcula_deltas_saida(saida_esperada)
		for i in range(1, n_cam + 1): # vai de 1 a n
			self.camadas[n_cam - i - 1].atualiza_pesos(self.camadas[n_cam - i]) # a penultima camada se atualiza baseada na ultima e assim por diante
			# quem atualiza a ultima camada????

	def treina_epoca(self, entradas): 
		for e in entradas:
			self.processa((e[1],e[2],e[3]))
			self.backpropagation(saidas_esperadas[e[5] - 1])

def normaliza(entradas):
	for e in entradas:
		e[1] = (e[1] - MIN_QPA) / (MAX_QPA - MIN_QPA)
		e[2] = (e[2] - MIN_PUL) / (MAX_PUL - MIN_PUL)
		e[3] = (e[3] - MIN_RSP) / (MAX_RSP - MIN_RSP)
	return entradas

def main():
	mlp = MLP()

	mlp.add_camada(Camada(N_ENT_C1, N_NEU_C1))
	mlp.add_camada(Camada(N_ENT_C2, N_NEU_C2))
	mlp.add_camada(Camada(N_ENT_C3, N_NEU_C3))
	mlp.add_camada(Camada(N_ENT_C3, N_NEU_C3)) #camada de saida
	#cam_saida = mlp.camadas[len(mlp.camadas) - 1]
	#for n in cam_saida.neuronios:
		#n.pesos[0] = 0.0
		#for i in range(1, len(n.pesos)):
			#n.pesos[i] = 1.0 
	
	f = open('data/02_treino_sinais_vitais_com_label.txt', 'r', encoding='UTF-8')
	dados = ent.lerEntradas(f, 0, N_ENTRADAS_TREINO)
	f.close()

	dados = normaliza(dados)

	print("-=- Treinando... -=-")
	for e in range(N_EPOCAS):
		mlp.treina_epoca(dados)
	print("-=- Treinado -=- \n")

	f = open('data/02_treino_sinais_vitais_com_label.txt', 'r', encoding='UTF-8')
	dados_val = ent.lerEntradas(f, N_ENTRADAS_TREINO, (N_ENTRADAS - N_ENTRADAS_TREINO))
	f.close()

	dados_val = normaliza(dados_val)

	for e in dados_val:
		saida = mlp.processa((e[1],e[2],e[3]))
		print(f"entrada: [{e[1]}, {e[2]}, {e[3]}]")
		print(f" classe esperada: {e[5]}")
		print(f" saida: [{round(saida[0],4)}, {round(saida[1],4)}, {round(saida[2],4)}, {round(saida[3],4)}] \n ")
		input("")

if __name__ == "__main__":
	main()

