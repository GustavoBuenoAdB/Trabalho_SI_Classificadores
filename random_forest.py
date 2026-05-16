from entrada import lerLinha, lerEntradas 
from math import log

N_ENTRADAS = 6
N_ARVORES = 10


#Discretando entradas
# - qPA

discretador = [[3, -4, 0, 4], [5, 20, 80, 140, 200], [3, 7, 15, 22]]


PA_BAIXA = -4 # entre -10 e -4
PA_MEDIA = 0 # entre -3 e 3
PA_ALTA = 4 # entre 4 e 10

# Pulso
PL_BAIXISSIMO = 20 #entre 0 e 20
PL_BAIXO = 80 #entre 21 e 80
PL_MEDIO = 140 # entre 81 e 140
PL_ALTO = 160 #entre 141 e 160
PL_ALTISSIMO = 200 #entre 161 e 200

#Respiração
RSP_BAIXA= 7 #entre 0 e 7
RSP_OK= 15 #entre 7 e 15
RSP_ALTA = 22 # entre 15 e 22		

# i_disc: 0 = qPA ; 1 = Pulso ; 2 = Respiração
def calc_entropia(i_disc, entradas):

	n_sub_grupos = discretador[i_disc][0] # discretador define os a quantidade e os limites de cada subgrupo

	labels = []
	for i in range(n_sub_grupos):
		labels.append([0, 0, 0, 0])

	# pra cada entrada, atualiza o histograma daquela classificação na label de cada subgrupo
	for e in entradas:
		for i in range(n_sub_grupos):
			if (e[i_disc + 1] > discretador[i_disc][i]):
				labels[i][e[5] - 1] += 1

	# soma a entropia de todos os subgrupos
	entropia = 0
	for i in range(n_sub_grupos):
		for j in range(4):
			prob = (labels[i][j] / len(labels[i]))
			entropia += prob * log((1/prob), 2)

	return entropia

def calc_mnr_entropia(flags_atrib, entradas):

	n_sub_grupos = []
	for i in range(3):
		if flags_atrib[i] == 1:
			n_sub_grupos.append(discretador[i][0]) # discretador define os a quantidade e os limites de cada subgrupo
		else:
			n_sub_grupos.append(0)

	hist_labels = []
	for i in range(3):
		if (n_sub_grupos[i] == 0):
			hist_labels.append([0]) #label de padding só
		for j in range(n_sub_grupos[i]):
			hist_labels.append([0, 0, 0, 0])

	# pra cada entrada, atualiza o histograma daquela classificação na label de cada subgrupo
	for e in entradas:
		for i in range(3): # for dos atributos
			aumnt = 0
			for j in range(0, n_sub_grupos[i][0]): # for dos subgrupos de cada atributo
				if (e[i + 1] < discretador[i][j + 1]):
					aumnt = j # flag de qual subgrupo incrementar
				hist_labels[aumnt][e[5] - 1] += 1 #aumenta o subgrupo na label tal

	# soma a entropia de todos os subgrupos de todos os atrubutos
	entropias = []
	for i in range(3): # for dos subatributos
		for j in range(n_sub_grupos[i][0]):
			total = 0
			for k in range(4):
				total += hist_labels[i][k] #total do subgrupo
			for k in range(4):
				prob = (hist_labels[i][k] / total) #pob de cada label
				entropias[i] += prob * log((1/prob), 2) #entropia daquele atributo

	min = 10000000000 #infinito
	ret = -1
	if min > entropias[0] and flags_atrib[0] == 1:
		min = entropias[0]
		ret = 0
	if min > entropias[1] and flags_atrib[1] == 1:
		min = entropias[1]
		ret = 1
	if min > entropias[2] and flags_atrib[2] == 1:
		min = entropias[2]
		ret = 2

	return ret

def separa_por_atributo(i_disc, entradas):

	n_sub_grupos = discretador[i_disc][0]

	# cria uma lista para cada subgrupo
	sub_grupos = []
	for i in range(n_sub_grupos):
		sub_grupos.append([]) # lista vazia

	# adiciona uma entrada em cada subgrupo respectivo
	for e in entradas:
		subg = 0
		for j in range(0, n_sub_grupos): # for dos subgrupos de cada atributo
			if (e[i_disc + 1] < discretador[i_disc][j + 1]):
				subg = j # flag de qual subgrupo adicionar
			sub_grupos[subg].append(e) 

	return sub_grupos

def calc_prob_labels(entradas):
	labels = [0,0,0,0]
	total = len(entradas)
	if total == 0:
		return (0, 0, 0, 0)

	for e in entradas:
		labels[e[5 - 1]] += 1
	
	return (labels[0] / total, labels[1] / total, labels[2] / total, labels[3] / total)

class No:
	def __init__(self, i_sub_atr):
		self.i_sub_atr = i_sub_atr # discretador define os a quantidade e os limites de cada subgrupo
		self.filhos = []
		self.prob = []

	def add_filho(self, no):
		self.filhos.append(no)

class Arvore:
	def __init__(self, n_atributos, entradas):

		flags = [1, 1, 1] #todas inicialmente ativas

		indx = calc_mnr_entropia(flags, entradas)

		self.raiz = No(indx)

		flags[indx] = 0 # abaixa a flag ja usada

		sub_grupos = separa_por_atributo(indx, entradas)

		for sg in sub_grupos:
			indx = calc_mnr_entropia(flags, sg)
			ramo = No(indx)
			flags[indx] = 0 # abaixa a flag ja usada

			ind_falta = 0 #pega o atributo que sobrou 
			for i in range(3):
				if flags[i] == 1:
					ind_falta = flags[i]

			flags[indx] = 1 #manter pro for

			sub_sub_grupos = separa_por_atributo(indx, entradas)

			# adiciona os filhos do ramo com probs calculadas
			for ssg in sub_sub_grupos:
				filho = No(ind_falta)
				filho.prob = calc_prob_labels(ssg)
				ramo.add_filho(filho)

			self.raiz.add_filho(No(indx))

		# escolhe uma hierarquia de atributos usando a entropia das entradas
		# cria a arvore baseada nesta hierarquia

	def processa(self, entrada):
		# só descer raiz abaixo e ver se resulta em True ou False e retorna esse voto.

class Floresta:
	def __init__(self):
		self.arvores = []
		# cria uma penca de arvores, (N_ARVORES)
	
	def processa(self, entrada):
		# manda cada arvore processar aquela entrada
		# contabiliza os votos e retorna

def main():
	# cria uma floresta, manda processar entradas 
	return 0


if __name__ == "__main__":
	main()