import entrada as ent
import random
import numpy
from scipy.special import expit #exipit é a sigmoid

N_ENTRADAS = 1499 # [1, 1500] = 1499
N_ENTRADAS_TREINO = 1000
N_EPOCAS = 10

TAM_MINI = 10 #queisso???
TAXA_APRENDIZAGEM = 0.02

MIN_PESO = 0.1
MAX_PESO = 0.9

#numeros de entrada e saidas de cada camada
N_ENT_C1 = 3
N_NEU_C1 = 4

N_ENT_C2 = 4
N_NEU_C2 = 3

N_ENT_C3 = 3
N_NEU_C3 = 1

#flags de "debug" = print
DEBUG_CAM = False
DEBUG_NEU = False
DEBUG_TRN = False

#normalizações
MIN_QPA = -10
MAX_QPA = 10
MIN_PUL = 0
MAX_PUL = 200
MIN_RSP = 0
MAX_RSP = 22
MIN_GRA = 0
MAX_GRA = 100


saidas_esperadas = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]] #alvos pra treino do back

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
		self.ult_z = 0 #essa é a ultima soma de pesos vezes entradas, antes da sigmoid
		self.ult_a = 0
		self.delta = 0 # derivada de Custo em relação a derivada de ativação da camada.
		self.taxa_apr = taxa_aprendizado

		#self.ultima_ativacao = [] #essa é a ultima saida?
		#self.ultima_som = [] #antes da sigmoid?
		
		# inicializa os pesos aleatoriamente
		for i in range (n_entradas + 1):
			self.pesos.append(random.uniform(0.1, 0.9))

		for i in range(self.n_entradas):
			self.ult_ent.append(0.0)

	def processa(self, entradas):
		
		som = 0
		n = len(entradas)
		for i in range(1, n + 1):
			som += entradas[i-1] * self.pesos[i]		
		som -= self.pesos[0] # esse aqui é o vies (indice 0 do vetor de pesos)
		ativacao = f_ativacao(som)

		#armazenando os valores do processamento
		for i in range(self.n_entradas):
			self.ult_ent[i] = entradas[i]
		self.ult_z = som
		self.ult_a = ativacao

		if (DEBUG_NEU):
			print(f" - soma: {som} \n - pesos: {self.pesos}")

		return ativacao
	
	def atualiza_pesos(self, atualizacao): #OBS: taxa pode virar parametro interno se nois for adicionar MOMENTUM
		if (DEBUG_TRN):
			print(f"- pesos......: {self.pesos}")
			print(f"- atualização: {atualizacao}")

		for i in range(self.n_entradas):
			self.pesos[i] += atualizacao[i]
			
			#normalizando os pesos
			'''f (self.pesos[i] < MIN_PESO):
				self.pesos[i] = MIN_PESO
			elif(self.pesos[i] > MAX_PESO):
				self.pesos[i] = MAX_PESO'''

		if (DEBUG_TRN):
			print(f"- pesos_atualizados: {self.pesos}\n")
	
class Camada:
	def __init__(self, n_entradas, n_saidas):
		
		self.n_entradas = n_entradas
		self.n_saidas = n_saidas #que tbm é o numero de neuronios
		self.neuronios = []

		# inicializa os neuronios da camada
		for i in range(n_saidas):
			self.neuronios.append(Neuronio(self.n_entradas, TAXA_APRENDIZAGEM))

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

		for n in self.neuronios:
			atualizacao = []

			# attualização do vies do neuronio n
			erro_propagado_vies = 0
			for n_prox in prox_camada.neuronios:
				c = 1*(derivada_sigmoid(n_prox.ult_z) * n_prox.delta) 
				erro_propagado_vies += c
				
			atualizacao.append((n.taxa_apr * n.ult_a * erro_propagado_vies)) # [0] do vetor de pesos

			# atualização de cada peso do neuronio n
			n.delta = 0 #erro propagado por n para a proxima camada
			for n_prox in prox_camada.neuronios:
				n.delta += (n.ult_a) * (derivada_sigmoid(n_prox.ult_z)) * n_prox.delta 
				
			for j in range(len(n.pesos) - 1): 
				atualizacao.append(-1*(n.taxa_apr * (n.pesos[j + 1] * n.ult_a * n.delta)))

			n.atualiza_pesos(atualizacao)

	def calcula_deltas_saida(self, saida_esperada):
		self.neuronios[0].delta = derivada_sigmoid(self.neuronios[0].ult_z) * (self.neuronios[0].ult_a - (saida_esperada[0]))

	
class MLP:
	def __init__(self):
		self.camadas = []

	def add_camada(self, camada):
		self.camadas.append(camada)

	def processa(self, entrada):
		saida = None
		for i in range(len(self.camadas) - 1):
			if (DEBUG_CAM):
				print(f"camada {i}")
			saida = (self.camadas[i].processa(entrada)) # enquanto tiver camada a saida de um é a entrada dotro
			entrada = saida

		self.camadas[-1].processa(entrada) 
		return saida
	
	def backpropagation(self, saida_esperada):
		n_cam = len(self.camadas)

		self.camadas[n_cam - 1].calcula_deltas_saida(saida_esperada)
		for i in range(1, n_cam): # vai de [1 a n[   n-1 termos
			if (DEBUG_TRN):
				print(f"=== Treinando camada {n_cam - i - 1} baseada na camada {n_cam - i}")

			self.camadas[n_cam - i - 1].atualiza_pesos(self.camadas[n_cam - i])

	def treina_epoca(self, entradas): 
		for e in entradas:
			self.processa((e[1],e[2],e[3]))
			self.backpropagation([e[4]])

def normaliza(entradas):
	for e in entradas:
		e[1] = (e[1] - MIN_QPA) / (MAX_QPA - MIN_QPA)
		e[2] = (e[2] - MIN_PUL) / (MAX_PUL - MIN_PUL)
		e[3] = (e[3] - MIN_RSP) / (MAX_RSP - MIN_RSP)
		e[4] = (e[4] - MIN_GRA) / (MAX_GRA - MIN_GRA)
	return entradas

def main():
	mlp = MLP()

	mlp.add_camada(Camada(N_ENT_C1, N_NEU_C1))
	mlp.add_camada(Camada(N_ENT_C2, N_NEU_C2))
	mlp.add_camada(Camada(N_ENT_C3, N_NEU_C3))
	cam_saida = Camada(N_NEU_C3, N_NEU_C3)
	for n in cam_saida.neuronios:
		n.pesos[0] = 0.0
		for i in range(1, len(n.pesos)):
			n.pesos[i] = 1.0 
	mlp.add_camada(cam_saida)
	
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
		print(f" saida esperada: {e[4]}")
		print(f" saida: {saida} \n ")
		input("")

if __name__ == "__main__":
	main()

