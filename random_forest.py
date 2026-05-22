import entrada as ent
from math import log
import time

N_ENTRADAS = 1499
N_ENTRADAS_TREINO = 1200
N_ENTRADAS_VALIDACAO = N_ENTRADAS - N_ENTRADAS_TREINO
N_ARVORES = 12
FLAG_BOOST = False # mais que 20 arvores é melhor essa que a otra
FLAG_BOOST_TODA_FLORESTA = True
MULTIPLICADOR = 2
TRSH_ADA = 0.5


VALIDACAO_CRUZADA = True

ARQUIVO = 'data/02_treino_sinais_vitais_com_label.txt'

#ids de acesso as entradas
ID_QPA = 1
ID_PUL = 2
ID_RSP = 3
ID_GRV = 4
ID_LAB = 5

#indice de nó que indica fim da arvore
ID_FOLHA = 20

DEBUB_VOT = False
DEBUG_TAM_ENTR = False

#Discretando entradas
# - qPA
N_GRUPOS_PA = 3
#limites superiores de cada grupo
PA_BAIXA = -2 # entre [-10 e -3[
PA_BOA = 2 # entre [-4 e 4[
PA_ALTA = 10 # entre [4 e 10]

# Pulso
N_GRUPOS_PL = 4
#limites superiores de cada grupo
PL_BAIXISSIMO = 40 #entre [0 e 20[
PL_BAIXO = 120 #entre [20 e 40[
PL_MEDIO = 160 # entre [40 e 140[
PL_ALTO = 200 #entre [140 e 160[

PL_ALTISSIMO = 200 #entre [160 e 200]

#Respiração
N_GRUPOS_RSP = 3
#limites superiores de cada grupo
RSP_BAIXA= 10 #entre [0 e 7[
RSP_OK= 20 #entre [7 e 15[
RSP_ALTA = 22 # entre [15 e 22[		

# inicializando o discretador de grupos
discretador = []
discretador.append([N_GRUPOS_PA, PA_BAIXA, PA_BOA, PA_ALTA])
discretador.append([N_GRUPOS_PL, PL_BAIXISSIMO, PL_BAIXO, PL_MEDIO, PL_ALTO, PL_ALTISSIMO])
discretador.append([N_GRUPOS_RSP, RSP_BAIXA, RSP_OK, RSP_ALTA])

def escolhe_atributo(flags_atrib, entradas):

	# inicializando uma  lista com os tamanhos de cada subgrupo
	n_sub_grupos = []
	for i in range(3):
		if flags_atrib[i] == 1:
			n_sub_grupos.append(discretador[i][0])
		else:
			n_sub_grupos.append(0)

	hist_labels = []
	for i in range(3):
		#cria um histograma pra cada subdivisão de cada grupo de cada atributo
		hist = []
		for j in range(n_sub_grupos[i]):
			hist.append([0, 0, 0, 0])
			
		hist_labels.append(hist)
	
	#somatorios de cada grupo pra talvez usar media e variancia para escolher
	som_grav_grupos = []
	for i in range(3):
		#cria um somatorio pra cada subdivisão de cada grupo de cada atributo
		som = []
		for j in range(n_sub_grupos[i]):
			som.append(0.0)
			
		som_grav_grupos.append(som)

	# pra cada entrada, atualiza o histograma e soma daquela classificação na label de cada subgrupo
	for e in entradas:
		for i in range(3): # for dos atributos
			if (flags_atrib[i] == 1):
				sub_grupo = escolhe_sub_grupo(i, e[i + 1])

				hist_labels[i][sub_grupo][e[ID_LAB] - 1] += 1
				som_grav_grupos[i][sub_grupo] += e[ID_GRV] 

	# soma a entropia de todos os subgrupos de todos os atributos
	entropias = [0, 0, 0]
	for i in range(3): # for dos atributos
		for j in range(n_sub_grupos[i]): #  for cada subgrupo
			if (flags_atrib[i] == 1):
				total = 0
				for k in range(4):
					total += hist_labels[i][j][k] #total do subgrupo
				for k in range(4):
					if (hist_labels[i][j][k] > 0):
						prob = (hist_labels[i][j][k] / total) #pob de cada label
						entropias[i] += prob * log((1/prob), 2) #entropia daquela label naquele subgrupo daquele atributo

	# escolhe o menor pra retornar
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

def separa_por_atributo(id_atrib, entradas):

	n_sub_grupos = discretador[id_atrib][0]

	# cria uma lista para cada subgrupo
	sub_grupos = []
	for i in range(n_sub_grupos):
		sub_grupos.append([]) # lista vazia

	# adiciona uma entrada em cada subgrupo respectivo
	for e in entradas:
		subg = escolhe_sub_grupo(id_atrib, e[id_atrib + 1])
		sub_grupos[subg].append(e) 

	return sub_grupos

def calc_voto_arv(entradas):
	labels = [0, 0, 0, 0]
	total = len(entradas)
	if total == 0:
		return (0.0, 0.0, 0.0, 0.0) , (0.0)

	som = 0.0
	for e in entradas:
		labels[e[ID_LAB] - 1] += 1
		som += e[ID_GRV]
	
	# calcula a prob de cada label e a média das gravidades
	return ((labels[0] / total), (labels[1] / total), (labels[2] / total), (labels[3] / total)) , (som / total)

def escolhe_sub_grupo(id_atrb, valor):
	n_grupos = discretador[id_atrb][0]
	ret = 0
	for i in range(0, n_grupos - 1):
		if (valor >= discretador[id_atrb][i + 1]): # +1 pq 0 é len
			ret = i + 1 # flag de qual filho seguir
	return ret 

class No:
	def __init__(self, i_sub_atr):
		self.i_sub_atr = i_sub_atr # discretador define os a quantidade e os limites de cada subgrupo
		self.filhos = []
		self.prob = [] #probabilidade de cada label
		self.grav = 0.0 #media das gravidades acumuladas

	def add_filho(self, no):
		self.filhos.append(no)

class Arvore:
	def __init__(self, entradas):

		flags = [1, 1, 1] #todas inicialmente ativas
		indx = escolhe_atributo(flags, entradas)

		self.raiz = No(indx)
		self.raiz.prob , self.raiz.grav = calc_voto_arv(entradas)
		flags[indx] = 0 # abaixa a flag ja usada

		sub_grupos = separa_por_atributo(indx, entradas)

		for sg in sub_grupos:
			indx = escolhe_atributo(flags, sg)
			ramo = No(indx)
			ramo.prob , ramo.grav = calc_voto_arv(entradas)

			flags[indx] = 0 # abaixa a flag ja usada
			ind_falta = 0 #pega o atributo que sobrou 
			for i in range(3):
				if flags[i] == 1:
					ind_falta = i
			flags[indx] = 1 #manter pro for

			sub_sub_grupos = separa_por_atributo(indx, sg)

			# adiciona os filhos do ramo com probs calculadas
			for ssg in sub_sub_grupos:
				filho = No(ind_falta)
				filho.prob , filho.grav = calc_voto_arv(entradas)

				folhas_grupos = separa_por_atributo(ind_falta, ssg)

				for fg in folhas_grupos:

					folha = No(ID_FOLHA) 
					folha.prob, folha.grav = calc_voto_arv(fg)
					#print(f"{folha.prob}, {folha.grav}")
					filho.add_filho(folha)

				ramo.add_filho(filho)

			self.raiz.add_filho(ramo)

	def processa(self, entrada):
		no_at = self.raiz
		
		prob_ret = no_at.prob
		grav_ret = no_at.grav

		# desce a arvore até chegar em uma folha
		while no_at.i_sub_atr != ID_FOLHA:

			atributo = no_at.i_sub_atr
			valor = entrada[atributo + 1] # +1 pra ignorar o id

			# descobre pra qual filho descer
			filho = escolhe_sub_grupo(atributo, valor)

			no_at = no_at.filhos[filho] 

			if (no_at.grav != 0.0):
				prob_ret = no_at.prob
				grav_ret = no_at.grav

		return prob_ret, grav_ret#retorna a o voto deste classificador

class Floresta:
	def __init__(self, n_arvores, entradas):
		n_ent = len(entradas)
		n_ent_p_arv = n_ent // n_arvores 
		self.arvores = []
		# inicializa n arvores dividindo igualmente as entradas
		arv_ant = Arvore(entradas[0 : n_ent_p_arv])
		self.arvores.append(arv_ant)

		for i in range(1, n_arvores):
			entr = entradas[(n_ent_p_arv * i) : (n_ent_p_arv * (i + 1))]

			if (FLAG_BOOST_TODA_FLORESTA):
				for a in self.arvores:
					entr = self.adaboost(entr, a, 2)
			if (FLAG_BOOST):
				a = self.arvores[-1]
				entr = self.adaboost(entr, a, 2)
			
			if (DEBUG_TAM_ENTR):
				print(len(entr))

			arv = Arvore(entr)
			self.arvores.append(arv)
			if (DEBUG_TAM_ENTR):
				print(f"arvore {i} criada")
	
	def processa(self, entrada):
		prob_res = [0.0, 0.0, 0.0, 0.0]
		grav_res = 0.0

		j = 0
		for a in self.arvores:

			prob, grav_m = a.processa(entrada)
			#print(f"aqui é a prob: {round(prob[0],2)},{round(prob[1],2)},{round(prob[2],2)},{round(prob[3],2)} da arvore")
			for i in range(4):
				prob_res[i] += prob[i]
			grav_res += grav_m

			if (DEBUB_VOT):
				print(f"arvore {j} vota em: gravidade {grav_m} e probs {prob}")

			j+=1

		n_arv = len(self.arvores)
		for i in range(4):
			prob_res[i] /= n_arv
		grav_res /= n_arv

		return prob_res, grav_res

	def adaboost(self, entradas, arv_ant, multiplicador):
		entradas_boost = [] 
		
		for e in entradas:
			prob, grav = arv_ant.processa(e)
			entradas_boost.append(e)

			if (prob[e[ID_LAB] - 1] > TRSH_ADA):
				for i in range(multiplicador - 1):
					entradas_boost.append(e)

		return entradas_boost

def main():

	for n_arvores_sai in range(1, 16):

		n_arvores_sai += 1

		print(f"n_arvores: {n_arvores_sai}")

		if VALIDACAO_CRUZADA:

			#cumuladores de coisa
			som_acuracia = 0
			som_precision = 0
			som_recall = 0
			som_f1 = 0

			som_tempo = 0

			for i in range((N_ENTRADAS) // (N_ENTRADAS_VALIDACAO)):

				tempo_inicio = time.time() * 1000 

				print(f"validação {i} de {(N_ENTRADAS) // (N_ENTRADAS_VALIDACAO)}")

				file = open(ARQUIVO,'r',encoding='UTF-8')
				treino = ent.lerEntradas(file, (N_ENTRADAS_VALIDACAO * i), N_ENTRADAS_TREINO)
				file.close()

				print("-=- Treinando... -=-")
				flor = Floresta(n_arvores_sai, treino)
				print("-=- Treino Finalizado -=-")

				file = open(ARQUIVO,'r',encoding='UTF-8')
				validacao = ent.lerEntradas(file, (N_ENTRADAS_VALIDACAO * i) + N_ENTRADAS_TREINO, (N_ENTRADAS_VALIDACAO))
				file.close()

				vp = 0
				fp = 0
				vn = 0
				fn = 0

				for e in validacao:
					prob, grav = flor.processa(e)

					#estado Critico
					if (e[ID_LAB] == 1):
						if (max(prob) != prob[0]):
							fp += 1 #falou que ta bem estando mal
						else:
							vn +=1
					
					#estado Instavel
					if (e[ID_LAB] == 2):
						if (max(prob) != prob[1]):
							fp += 1 #falou que ta bem estando mal
						else:
							vn +=1

					#estado Potencialmente Estavel
					if (e[ID_LAB] == 3):
						if (max(prob) != prob[2]):
							fn += 1 #falou que ta mal estando bem
						else:
							vp +=1

					#estado Estavel
					if (e[ID_LAB] == 4):
						if (max(prob) != prob[3]):
							fn += 1 #falou que ta mal estando bem
						else:
							vp +=1

					#print(f"Entrada: gravidade {e[ID_GRV]}, label {e[ID_LAB]}")
					#print(f"Saida..: gravidade {grav}, probs {prob[e[ID_LAB] - 1]}")
					#input("")

				if (vp + vn + fp + fn) != 0:
					acuracia = (vp + vn) / (vp + vn + fp + fn)
					som_acuracia += acuracia
					#print(f"acuracia: {round(acuracia, 4) * 100} %")
				if (vp + fp) != 0:
					precisao = (vp) / (vp + fp)
					som_precision += precisao
					#print(f"precisao: {round(precisao, 4) * 100} %")

				if (vp + fn) != 0:
					recall = (vp) / (vp + fn)
					som_recall += recall
					#print(f"recall..: {round(recall, 4) * 100} %")

				if precisao != 0 and recall != 0:
					f1 = 2*((precisao * recall) / (precisao + recall))
					som_f1 += f1
					#print(f"f1......: {round(f1, 4) * 100} %")

				tempo_final = time.time() * 1000
				som_tempo += (tempo_final - tempo_inicio) 

			som_acuracia /= (N_ENTRADAS) // (N_ENTRADAS_VALIDACAO)
			som_precision /= (N_ENTRADAS) // (N_ENTRADAS_VALIDACAO)
			som_recall /= (N_ENTRADAS) // (N_ENTRADAS_VALIDACAO)
			som_f1 /= (N_ENTRADAS) // (N_ENTRADAS_VALIDACAO)

			som_tempo /= (N_ENTRADAS) // (N_ENTRADAS_VALIDACAO)

			x = som_acuracia
			y = som_precision
			z = som_recall
			a = som_f1

			#arquivo_saida = open("acu_pre_rec_f1_n_arv_sem_ada_1a15.csv", "a", encoding="utf-8")
			#print(f"{x},{y},{z},{a}", file=arquivo_saida)

			arquivo_saida_2 =  open("tempo_n_arv_com_ada_mult_1a15.csv", "a", encoding="utf-8")
			print(f"{som_tempo}", file=arquivo_saida_2)

	else:
		file = open(ARQUIVO,'r',encoding='UTF-8')
		treino = ent.lerEntradas(file, 0, N_ENTRADAS_TREINO)
		file.close()

		print("-=- Treinando... -=-")
		flor = Floresta(N_ARVORES, treino)
		print("-=- Treino Finalizado -=-")

		file = open(ARQUIVO,'r',encoding='UTF-8')
		validacao = ent.lerEntradas(file, N_ENTRADAS_TREINO, (N_ENTRADAS - N_ENTRADAS_TREINO))
		file.close()

		vp = 0
		fp = 0
		vn = 0
		fn = 0

		for e in validacao:
			prob, grav = flor.processa(e)

			#estado Critico
			if (e[ID_LAB] == 1):
				if (max(prob) != prob[0]):
					fp += 1 #falou que ta bem estando mal
				else:
					vn +=1
			
			#estado Instavel
			if (e[ID_LAB] == 2):
				if (max(prob) != prob[1]):
					fp += 1 #falou que ta bem estando mal
				else:
					vn +=1

			#estado Potencialmente Estavel
			if (e[ID_LAB] == 3):
				if (max(prob) != prob[2]):
					fn += 1 #falou que ta mal estando bem
				else:
					vp +=1

			#estado Estavel
			if (e[ID_LAB] == 4):
				if (max(prob) != prob[3]):
					fn += 1 #falou que ta mal estando bem
				else:
					vp +=1

			#print(f"Entrada: gravidade {e[ID_GRV]}, label {e[ID_LAB]}")
			#print(f"Saida..: gravidade {grav}, probs {prob[e[ID_LAB] - 1]}")
			#input("")

		if (vp + vn + fp + fn) != 0:
			acuracia = (vp + vn) / (vp + vn + fp + fn)
			print(f"acuracia: {round(acuracia, 4) * 100} %")
		if (vp + fp) != 0:
			precisao = (vp) / (vp + fp)
			print(f"precisao: {round(precisao, 4) * 100} %")

		if (vp + fn) != 0:
			recall = (vp) / (vp + fn)
			print(f"recall..: {round(recall, 4) * 100} %")

		if precisao != 0 and recall != 0:
			f1 = 2*((precisao * recall) / (precisao + recall))
			print(f"f1......: {round(f1, 4) * 100} %")

		return 0

if __name__ == "__main__":
	main()