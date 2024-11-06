
import numpy as np

from perceptron import Perceptron


def main():
    caracteristicas=(['Experiência', 'Certificação', 'Recomendação/Conexões','Disponibilidade'])

    # Dados de treinamento: entradas e saídas desejadas
    amostra = np.array([

        [0, 0, 0, 0],   #0
        [0, 0, 1, 1],   #0
        [0, 1, 1, 0],   #0
        [1, 1, 0, 1],   #1
        [1, 0, 1, 1],   #1
        [1, 1, 0, 0],   #0
        [0, 1, 0, 1],   #1
        [0, 0, 1, 0],  # 0
        [0, 1, 0, 0],  # 0

    ])
    y = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0])  # Saídas desejadas: 1 para apto, 0 para não apto

    # Instanciar o perceptron com 4 entradas e uma taxa de aprendizado de 0.1
    perceptron_model = Perceptron(input_size=4, escala_aprendizagem=0.1)

    # Treinamento do perceptron
    print("Iniciando o treinamento do Perceptron...\n")
    perceptron_model.treinar(amostra, y, ciclos=1500)
    print("\nTreinamento concluído!\n")

    # Testar o modelo com novos exemplos
    novos_candidatos = np.array([

        [0, 1, 0, 1],  # 1
        [1, 1, 1, 1],  # 1
        [1, 0, 0, 0],  # 0
        [1, 0, 1, 0],  # 0
        [1, 0, 1, 1],  # 1  # Exemplo de candidato que está apto para ser entrevistado com certificação e disponibilidade
    ])

    print("Resultados para novos candidatos a vaga:")
    for i, candidato in enumerate(novos_candidatos):
        print(f"Candidato {i + 1}")
        for j, caracteristica in enumerate(caracteristicas):
            print(f"{caracteristica} => {candidato[j]}")
        resultado = perceptron_model.consolidacao(candidato)
        print(f"Resultado: {'Apto' if resultado == 1 else 'Não apto'}\n")


if __name__ == "__main__":
    main()