import entrada as ent
import random
import numpy
from scipy.special import expit #exipit é a sigmoid

N_ENTRADAS_TREINO = 50
N_EPOCAS = 5
TAM_MINI = 10
TAXA_APRENDIZAGEM = 0.01
#numeros de entrada e saidas de cada camada
N_ENT_C1 = 3
N_NEU_C1 = 4

N_ENT_C2 = 4
N_NEU_C2 = 6

N_ENT_C3 = 6
N_NEU_C3 = 4

#flags de "debug" = print
DEBUG_CAM = True
DEBUG_NEU = False

saida_esperada = [[1.00,0.00,0.00,0.00],[0.00,1.00,0.00,0.00],[0.00,0.00,1.00,0.00],[0.00,0.00,0.00,1.00]]

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
		self.ultima_ativacao = []
		self.ultima_som = []
		self.vies = random.random()
		# inicializa os pesos aleatoriamente
		for i in range (n_entradas + 1):
			self.pesos.append(random.uniform(-0.5, 0.5))


	def processa(self, entradas):
		som = 0
		n = len(entradas)
		for i in range(1, n + 1):
			som += entradas[i-1] * self.pesos[i]		
		som += self.pesos[0] # esse aqui é o vies (indice 0 do vetor de pesos)
		self.ultima_som = som
		self.ultima_entrada = entradas
		if (DEBUG_NEU):
			print(f" - entrada: {entradas} \n - pesos: {self.pesos} \n - soma: {som}")
		ativacao = f_ativacao(som)
		self.ultima_ativacao = ativacao #guardando a anterior para utlilzat no backprop
		return ativacao
	
	def atualiza_pesos(self, delta, taxa):
		self.pesos[0] = self.pesos[0] - taxa * delta  # atualiza vies
		for i in range(self.n_entradas):
			self.pesos[i+1] -= taxa * delta * self.ultima_entrada[i] #pq i -1? AUGUSTO TODO

	
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
		for i, neuronio in enumerate(self.neuronios):
			erro = 0
			for j, neuronio_frente in enumerate(camada_frente.neuronios):
				erro += deltas_frente[j] * neuronio_frente.pesos[i+1]
			
			delta = erro * derivada_sigmoid(neuronio.ultima_som)
			neuronio.atualiza_pesos(delta, taxa)
			deltas.append(delta)
		
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
		saida = entrada
		for i in range(len(self.camadas)):
			if (DEBUG_CAM):
				print(f"camada {i}")
			saida = (self.camadas[i].processa(saida)) # enquanto tiver camada a saida de um é a entrada dotro

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
	def uma_epoca(self,saida,esperado):
		deltas = self.camadas[2].backprop_saida(saida, esperado, TAXA_APRENDIZAGEM)
		print(deltas)
		for i in range(len(self.camadas)-2,0,-1):
			deltas = self.camadas[i].backprop(deltas,self.camadas[i+1],TAXA_APRENDIZAGEM)
	
	def treina(self,):
		#aqui precisa orquestrar o treinamento em cada mini pedaço de treino para cada época
		return 1


def main():
	mlp = MLP()

	mlp.add_camada(Camada(N_ENT_C1, N_NEU_C1))
	mlp.add_camada(Camada(N_ENT_C2, N_NEU_C2))
	mlp.add_camada(Camada(N_ENT_C3, N_NEU_C3))
	print(f"Esses são os pesos iniciais: {mlp.camadas[1].neuronios[1].pesos}")
	
	f = open('data/02_treino_sinais_vitais_com_label.txt', 'r', encoding='UTF-8')
	dados = ent.lerEntradas(f, 0, N_ENTRADAS_TREINO)

	for i in dados:
		saida = mlp.processa((i[1],i[2],i[3]))
		print(f"classe esperada: {i[5]} | saida: {saida}")
		mlp.uma_epoca(saida,i[5]-1) #-1 para evitar que ele tente acessar o indice 4 quando a classe for 4 e quebre
	
if __name__ == "__main__":
	main()


# TODO: aqui ta implementada só a base da base, não uma_epoca
#nem pega de arquivo, ainda é bem xunxa, n sei inclusive se o vies o - la ta sendo calculado certinho
# pra ser sincero meio que esqueci dele, tem que ver isso, adicionar leitura dos dados certinhos la, e programar a tal da BACKPROPAGATION, mas aqui é só 
# o esquletinho pra brincar de ajustar parametro de camada