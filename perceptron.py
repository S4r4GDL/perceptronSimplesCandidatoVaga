import numpy as np


# produz resultados no intervalo entre 0 e 1, 
# mas ela força para que os resultados se concentrem mais próximos de 0 ou 1
# ou seja, aumenta a precisão no nosso perceptron binário
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Perceptron:

    def __init__(self, input_size, escala_aprendizagem=0.1):
        # Inicializar pesos e bias aleatoriamente
        self.pesos = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.escala_aprendizagem = escala_aprendizagem

    def soma_pesos(self, entradas):
        return np.dot(entradas, self.pesos) + self.bias

    def consolidacao(self, entradas):
        total_soma = self.soma_pesos(entradas)
        saida = sigmoid(total_soma)
        return 1 if saida >= 0.5 else 0  # Retorna 1 para apto, 0 para não apto

    def treinar(self, X, y, ciclos=100):
        for ciclo in range(ciclos):
            total_erros = 0
            for entradas, target in zip(X, y):

                # Passo 1: Calcular a saída do perceptron
                total_soma = self.soma_pesos(entradas)
                saida = sigmoid(total_soma)

                # Passo 2: Calcular o erro (diferença entre a saída desejada e a saída atual)
                erro = target - saida
                total_erros += erro ** 2  # Acumular o erro quadrado

                # Passo 3: Ajustar pesos e bias
                # Atualização dos pesos usando o gradiente descendente
                self.pesos += self.escala_aprendizagem * erro * saida * (1 - saida) * np.array(entradas)

                # Atualização do bias
                self.bias += self.escala_aprendizagem * erro * saida * (1 - saida)

            # Exibir o erro médio por época
            print(f"Ciclo {ciclo + 1}/{ciclos}, erro: {total_erros / len(X)}")
