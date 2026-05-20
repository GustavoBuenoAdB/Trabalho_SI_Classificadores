import entrada as ent

N_ENTRADAS = 1499 # [1, 1500] = 1499 

# limites para o histograma
LIM_INF_HIST_QPA = -10
LIM_SUP_HIST_QPA = 1

LIM_INF_HIST_PUL = 180
LIM_SUP_HIST_PUL = 200

LIM_INF_HIST_RSP = 16
LIM_SUP_HIST_RSP = 22

ID_QPA = 1
ID_PUL = 2
ID_RSP = 3
ID_LAB = 5

def main():
	f = open('data/02_treino_sinais_vitais_com_label.txt','r',encoding='UTF-8')
	entradas = ent.lerEntradas(f, 1, N_ENTRADAS)
	f.close()

	somas = [0.0, 0.0, 0.0]
	somas_p_label = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
	hist_labels = [0, 0, 0, 0]
	hist_labels_com_lim = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

	max_gravidade = 0
	min_gravidade = 100

	for e in entradas:
		#somatorios totais
		somas[0] += e[ID_QPA]
		somas[1] += e[ID_PUL]
		somas[2] += e[ID_RSP]

		# somatorios divididos por resultados esperados
		somas_p_label[e[ID_LAB] - 1][0] += e[ID_QPA]
		somas_p_label[e[ID_LAB] - 1][1] += e[ID_PUL]
		somas_p_label[e[ID_LAB] - 1][2] += e[ID_RSP]

		# aparições de cada label
		hist_labels[e[ID_LAB] - 1] += 1

		# apariçõa de cada label satisfazendo limites
		if LIM_INF_HIST_QPA <= e[ID_QPA] and e[ID_QPA] <= LIM_SUP_HIST_QPA:
			hist_labels_com_lim[0][e[ID_LAB] - 1] += 1

		if LIM_INF_HIST_PUL <= e[ID_PUL] and e[ID_PUL] <= LIM_SUP_HIST_PUL:
			hist_labels_com_lim[1][e[ID_LAB] - 1] += 1
		
		if LIM_INF_HIST_RSP <= e[ID_RSP] and e[ID_RSP] <= LIM_SUP_HIST_RSP:
			hist_labels_com_lim[2][e[ID_LAB] - 1] += 1

		if e[4] < min_gravidade:
			min_gravidade = e[4]
		if e[4] > max_gravidade:
			max_gravidade = e[4]

	# mostrando resultados
	print(f"Em {N_ENTRADAS} entradas: \n")

	print(f"Gravidade mínima: {min_gravidade}")
	print(f"Gravidade máxima: {max_gravidade}") 
	print("") #\n

	print("Médias:") 
	print(f" - qPA: {(somas[0] / N_ENTRADAS)}")
	print(f" - Pul: {(somas[1] / N_ENTRADAS)}")
	print(f" - Rsp: {(somas[2] / N_ENTRADAS)}")
	print("") #\n

	print("Médias por Label:") 
	for i in range(4):
		print(f"Médias para Label {i+1}:") 
		print(f" - qPA: {(somas_p_label[i][0] / hist_labels[i])}")
		print(f" - Pul: {(somas_p_label[i][1] / hist_labels[i])}")
		print(f" - Rsp: {(somas_p_label[i][2] / hist_labels[i])}")
		print("") #\n

	print("Distribuição das Entradas:")
	for i in range(4):
		print(f"Label {i+1}: {round(((hist_labels[i] / N_ENTRADAS) * 100),4)}% ") 
	print("") #\n

	print("Distribuição das Entradas, respeitando Limites:")
	print(f"Limites em qPA de [{LIM_INF_HIST_QPA}, {LIM_SUP_HIST_QPA}]")
	for i in range(4):
		print(f"Label {i+1}: {round(((hist_labels_com_lim[0][i] / N_ENTRADAS) * 100),4)}% ") 
	print("") #\n
	print(f"Limites em Pulso de [{LIM_INF_HIST_PUL}, {LIM_SUP_HIST_PUL}]")
	for i in range(4):
		print(f"Label {i+1}: {round(((hist_labels_com_lim[1][i] / N_ENTRADAS) * 100),4)}% ") 
	print("") #\n
	print(f"Limites em Respiração de [{LIM_INF_HIST_RSP}, {LIM_SUP_HIST_RSP}]")
	for i in range(4):
		print(f"Label {i+1}: {round(((hist_labels_com_lim[2][i] / N_ENTRADAS) * 100),4)}% ") 
	print("") #\n

if __name__ == "__main__":
	main()