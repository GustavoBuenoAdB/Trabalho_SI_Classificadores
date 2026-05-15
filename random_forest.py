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

def esc_ind_entropia(entradas):

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


def calc_vet_entropia(pai,n_atributos, entradas):
	
	e_qPA = calc_entrop_qPA(entradas)
	e_pulso = calc_entrop_pulso(entradas)
	e_RSP = calc_entrop_RSP(entradas)
	
	menosEntropico = max(e_qPA, e_pulso, e_RSP)
	
	if menosEntropico == e_qPA:
		#separo em 3 grupos por q_PA
		#confere cada subgrupo
		#separa tudo

		e_baixa = []
		e_media = []
		e_alta = []

		for e in entradas:
			if (e[1] < PA_BAIXA):
				e_baixa.append(e)
			elif (e[1] > PA_ALTA):
				e_alta.append(e)
			else:
				e_media.append(e)
		
		e_pulso = calc_entrop_pulso(e_baixa)
		e_RSP = calc_entrop_RSP(e_baixa)

		if (e_pulso > e_RSP):

			ee_baixissima = []
			ee_baixa = []
			ee_media = []
			ee_alta = []
			ee_altissima = []

			for e in e_baixa:
				if (e[1] < PL_BAIXISSIMO):
					ee_baixissima.append(e)
				elif (e[1] > PL_BAIXO and e[1] < PL_MEDIO):
					ee_baixa.append(e)
				elif (e[1] > PL_MEDIO and e[1] < PL_ALTO):
					ee_media.append(e)
				elif (e[1] > PL_ALTO and e[1] < PL_ALTISSIMO):
					ee_alta.append(e)
				else:
					ee_altissima.append(e)

				eee_baixa = []
				eee_media = []
				eee_alta = []

				for e in entradas:
					if (e[1] < RSP_BAIXA):
						eee_baixa.append(e)
					elif (e[1] > RSP_ALTA):
						eee_alta.append(e)
					else:
						eee_media.append(e)

def calc_entrop_qPA(entradas):
	baixa = 0
	media = 0
	alta = 0

	labels_b = [0, 0, 0, 0]
	labels_m = [0, 0, 0, 0]
	labels_a = [0, 0, 0, 0]

	for e in entradas:
		if (e[1] < PA_BAIXA):
			labels_b[e[5] - 1] += 1
			baixa += 1
		elif (e[1] > PA_ALTA):
			labels_a[e[5] - 1] += 1
			alta += 1
		else:
			labels_m[e[5] - 1] += 1
			media += 1

	entropia = 0

	for i in range(4):
		prob = (labels_b[i] / baixa)
		entropia += prob * log((1/prob),2)

	for i in range(4):
		prob = (labels_m[i] / media)
		entropia += prob * log((1/prob),2)

	for i in range(4):
		prob = (labels_a[i] / alta)
		entropia += prob * log((1/prob),2)

	return entropia

def calc_entrop_pulso(entradas):
	baixissima = 0
	baixa = 0
	media = 0
	alta = 0
	altissima = 0

	labels_bb = [0, 0, 0, 0]
	labels_b = [0, 0, 0, 0]
	labels_m = [0, 0, 0, 0]
	labels_a = [0, 0, 0, 0]
	labels_aa = [0, 0, 0, 0]

	for e in entradas:
		if (e[1] < PL_BAIXISSIMO):
			labels_b[e[5] - 1] += 1
			baixissima += 1
		elif (e[1] > PL_BAIXO and e[1] < PL_MEDIO):
			labels_b[e[5] - 1] += 1
			baixa += 1
		elif (e[1] > PL_MEDIO and e[1] < PL_ALTO):
			labels_m[e[5] - 1] += 1
			media += 1
		elif (e[1] > PL_ALTO and e[1] < PL_ALTISSIMO):
			labels_a[e[5] - 1] += 1
			alta += 1
		else:
			labels_aa[e[5] - 1] += 1
			altissima += 1

	entropia = 0

	for i in range(4):
		prob = (labels_bb[i] / baixissima)
		entropia += prob * log((1/prob),2)

	for i in range(4):
		prob = (labels_b[i] / baixa)
		entropia += prob * log((1/prob),2)

	for i in range(4):
		prob = (labels_m[i] / media)
		entropia += prob * log((1/prob),2)

	for i in range(4):
		prob = (labels_a[i] / alta)
		entropia += prob * log((1/prob),2)

	for i in range(4):
		prob = (labels_aa[i] / altissima)
		entropia += prob * log((1/prob),2)

	return entropia



def calc_entrop_RSP(entradas):
	baixa = 0
	ok = 0
	alta = 0

	labels_b = [0, 0, 0, 0]
	labels_o = [0, 0, 0, 0]
	labels_a = [0, 0, 0, 0]

	for e in entradas:
		if (e[1] < RSP_BAIXA):
			labels_b[e[5] - 1] += 1
			baixa += 1
		elif (e[1] > RSP_ALTA):
			labels_a[e[5] - 1] += 1
			alta += 1
		else:
			labels_o[e[5] - 1] += 1
			ok += 1

	entropia = 0

	for i in range(4):
		prob = (labels_b[i] / baixa)
		entropia += prob * log((1/prob),2)

	for i in range(4):
		prob = (labels_o[i] / ok)
		entropia += prob * log((1/prob),2)

	for i in range(4):
		prob = (labels_a[i] / alta)
		entropia += prob * log((1/prob),2)

	return entropia

class Arvore:
	def __init__(self, n_atributos, n_entradas, entradas):

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