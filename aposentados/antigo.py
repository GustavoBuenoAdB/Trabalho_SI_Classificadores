	'''
	def calcula_custo(entrada,saida):
		esperado = saida_esperada[entrada[N_NEU_C3]]
		custo = []
		for i in range(N_NEU_C3):
			custo.append((saida[i] - esperado[i])*(saida[i] - esperado[i])) 
		media_custo = sum(custo) / len(custo)
		return media_custo
	'''

	'''def backprop(self, deltas_frente, camada_frente, taxa):
		if (DEBUG_TRN):
			print("treino da camada anterior:")
		
		deltas = []
		for i, neuronio in enumerate(self.neuronios): #enumerates? len(self.neuronio) n funciona?
			erro = 0
			for j, neuronio_frente in enumerate(camada_frente.neuronios): # nesse for pega o erro que eu propaguei.
				erro += deltas_frente[j] * neuronio_frente.pesos[i+1] # o erro do proximo neuronio veiz o peso do que eu dei de entrada pra ele?
			
			delta = erro * derivada_sigmoid(neuronio.ultima_som) #aqui eu perdi, essa som é a ultima soma de erro? não, soma antes da sigmoid né.
			neuronio.atualiza_pesos(delta, taxa)
			deltas.append(delta) # guarda o erro de cada um pra prox camada, que é a anterior
		
		return deltas'''

	'''def backprop_saida(self, saida, esperado, taxa):
		if (DEBUG_TRN):
			print("treino da camada de saida:")
			print(f"saida esperada: {saida_esperada[esperado]}")
			print(f"saida obtida..: {saida}\n")
		
		deltas = []
		for i in range(N_NEU_C3):
			erro = pow((saida[i] - saida_esperada[esperado][i]), 2) # pq aqui o erro n é quadratico???
			delta = self.calcula_delta(erro, i)
			self.neuronios[i].atualiza_pesos(delta, taxa)
			deltas.append(delta)
		return deltas
	
	def calcula_delta(self, erro, indx):
		#delta = derivada do custo * derivada da sigmoide
		d_sigmoide = derivada_sigmoid(self.neuronios[indx].ultima_som) 
		delta = erro * d_sigmoide # esse caba é o que ???
		return delta'''

'''	def uma_epoca(self, saida, esperado): #bro, isso aqui é treino, epoca é cada ciclo completo que vc faz no conjunto de treino KSKSKSKSKSK confundiu se pa
		deltas = self.camadas[2].backprop_saida(saida, esperado, TAXA_APRENDIZAGEM)
		#print(deltas)
		for i in range(len(self.camadas)-2, -1, -1):
			deltas = self.camadas[i].backprop(deltas, self.camadas[i+1],TAXA_APRENDIZAGEM)

	def treina(self,):
		#aqui precisa orquestrar o treinamento em cada mini pedaço de treino para cada época
		return 1'''