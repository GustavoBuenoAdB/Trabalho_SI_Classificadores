import entrada as ent
import random
import numpy
from scipy.special import expit #exipit é a sigmoid

N_ENTRADAS_TREINO = 100
N_EPOCAS = 5
TAM_MINI = 10 #queisso???
TAXA_APRENDIZAGEM = 0.01
#numeros de entrada e saidas de cada camada
N_ENT_C1 = 3
N_NEU_C1 = 4

N_ENT_C2 = 4
N_NEU_C2 = 3

N_ENT_C3 = 3
N_NEU_C3 = 4

#flags de "debug" = print
DEBUG_CAM = False
DEBUG_NEU = True

#normalizações
MIN_QPA = -10
MAX_QPA = 10
MIN_PUL = 0
MAX_PUL = 200
MIN_RSP = 0
MAX_RSP = 22


saida_esperada = [[1.00,0.00,0.00,0.00],[0.00,1.00,0.00,0.00],[0.00,0.00,1.00,0.00],[0.00,0.00,0.00,1.00]] #alvos pra treino do back

def f_ativacao(x):
	return expit(x)

def derivada_sigmoid(x):
	s = f_ativacao(x)
	return s * (1 - s)

class Neuronio:
	def __init__(self, n_entradas):
		self.n_entradas = n_entradas
		self.pesos = []
		self.ultima_entrada = []
		self.ultima_ativacao = [] #essa é a ultima saida?
		self.ultima_som = [] #antes da sigmoid?
		self.vies = random.random()
		# inicializa os pesos aleatoriamente
		for i in range (n_entradas + 1):
			self.pesos.append(random.uniform(-0.5, 0.5))


	def processa(self, entradas):
		
		som = 0
		n = len(entradas)
		for i in range(1, n + 1):
			som += entradas[i-1] * self.pesos[i]		
		som -= self.pesos[0] # esse aqui é o vies (indice 0 do vetor de pesos)

		self.ultima_som = som
		self.ultima_entrada = entradas
		ativacao = f_ativacao(som)
		self.ultima_ativacao = ativacao #guardando a anterior para utlilzat no backprop

		if (DEBUG_NEU):
			print(f" - entrada: {entradas} \n - pesos: {self.pesos} \n - soma: {som}")
		return ativacao
	
	def atualiza_pesos(self, delta, taxa): #OBS: taxa pode virar parametro interno se nois for adicionar MOMENTUM
		self.pesos[0] = self.pesos[0] - taxa * delta  # atualiza vies
		for i in range(self.n_entradas):
			self.pesos[i+1] -= taxa * delta * self.ultima_entrada[i] #pq i -1? AUGUSTO TODO ??? q ??? acho q tu ja arrumou isso, ja que entrada é n+1 de pesos em tamanho.
			#pq - aqui? n é +???

	
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
	
	def backprop(self, deltas_frente, camada_frente, taxa):
		deltas = []
		for i, neuronio in enumerate(self.neuronios): #enumerates? len(self.neuronio) n funciona?
			erro = 0
			for j, neuronio_frente in enumerate(camada_frente.neuronios): # nesse for pega o erro que eu propaguei.
				erro += deltas_frente[j] * neuronio_frente.pesos[i+1] # o erro do proximo neuronio veiz o peso do que eu dei de entrada pra ele?
			
			delta = erro * derivada_sigmoid(neuronio.ultima_som) #aqui eu perdi, essa som é a ultima soma de erro? não, soma antes da sigmoid né.
			neuronio.atualiza_pesos(delta, taxa)
			deltas.append(delta) # guarda o erro de cada um pra prox camada, que é a anterior
		
		return deltas

	def backprop_saida(self, saida, esperado, taxa):
		deltas = []
		for i in range(N_NEU_C3):
			erro = saida[i] - saida_esperada[esperado][i]
			delta = self.calcula_delta(erro,i)
			self.neuronios[i].atualiza_pesos(delta, taxa)
			deltas.append(delta)
		return deltas
	
	def calcula_delta(self, erro, indx):
		#delta = derivada do custo * derivada da sigmoide
		d_sigmoide = derivada_sigmoid(self.neuronios[indx].ultima_som) 
		delta = erro * d_sigmoide
		return delta
	
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
	'''
	def calcula_custo(entrada,saida):
		esperado = saida_esperada[entrada[N_NEU_C3]]
		custo = []
		for i in range(N_NEU_C3):
			custo.append((saida[i] - esperado[i])*(saida[i] - esperado[i])) 
		media_custo = sum(custo) / len(custo)
		return media_custo
	'''
	def uma_epoca(self,saida,esperado): #bro, isso aqui é treino, epoca é cada ciclo completo que vc faz no conjunto de treino KSKSKSKSKSK confundiu se pa
		deltas = self.camadas[2].backprop_saida(saida, esperado, TAXA_APRENDIZAGEM)
		#print(deltas)
		for i in range(len(self.camadas)-2,0,-1):
			deltas = self.camadas[i].backprop(deltas,self.camadas[i+1],TAXA_APRENDIZAGEM)
	
	def treina(self,):
		#aqui precisa orquestrar o treinamento em cada mini pedaço de treino para cada época
		return 1

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
	print(f"Esses são os pesos iniciais: {mlp.camadas[1].neuronios[1].pesos}")
	
	f = open('data/02_treino_sinais_vitais_com_label.txt', 'r', encoding='UTF-8')
	dados = ent.lerEntradas(f, 0, N_ENTRADAS_TREINO)
	f.close()

	dados = normaliza(dados)

	for e in range(N_EPOCAS):
		print(f"epoca {e}")
		for i in dados:
			saida = mlp.processa((i[1],i[2],i[3]))

			print(f"\n entrada: [{i[1]}, {i[2]}, {i[3]}]") 
			print(f" classe esperada: {i[5]}") 
			print(f" saida: {saida} \n ")

			mlp.uma_epoca(saida,i[5]-1) #-1 para evitar que ele tente acessar o indice 4 quando a classe for 4 e quebre
	
if __name__ == "__main__":
	main()


# TODO: aqui ta implementada só a base da base, não uma_epoca
#nem pega de arquivo, ainda é bem xunxa, n sei inclusive se o vies o - la ta sendo calculado certinho
# pra ser sincero meio que esqueci dele, tem que ver isso, adicionar leitura dos dados certinhos la, e programar a tal da BACKPROPAGATION, mas aqui é só 
# o esquletinho pra brincar de ajustar parametro de camada