# Abordagem One-Shot Learning para Classificação de Imagens de Inspeções Submarinas

As inspeções submarinas são essenciais para a manutenção de infraestruturas
offshore, mas enfrentam desafios como alto custo de coleta, baixa qualidade
das imagens e escassez de dados rotulados. Devido à diversidade de
objetos e estruturas, modelos tradicionais são limitados, tornando técnicas
que aprendem com poucos exemplos relevantes e representativos. O presente
trabalho propõe um classificador de imagens voltado para inspeções submarinas
utilizando a abordagem de One-Shot Learning. Essa técnica permite que
modelos de aprendizado de máquina reconheçam novas classes a partir de um
número extremamente limitado de exemplos, superando a limitação de conjuntos
de dados escassos, comuns em aplicações submarinas. Para isso, foi
utilizada uma rede neural do tipo Siamesa, capaz de aprender uma função de
similaridade entre pares de imagens. O modelo foi treinado em um conjunto de
imagens de inspeções submarinas contendo diferentes categorias, como ROV,
dutos, flanges, manipuladores, objetos e equipamentos, e avaliado em tarefas
N-way, utilizando métricas como acurácia, precisão, recall e F1-score. Os resultados
demonstraram que a abordagem é eficaz na identificação de classes
inéditas com desempenho competitivo, evidenciando a aplicabilidade do One-
Shot Learning em cenários de inspeção submarina com disponibilidade limitada
de dados.
