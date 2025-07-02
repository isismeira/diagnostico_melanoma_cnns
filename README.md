# Classificação de sinais de pele para diagnóstico de melanoma através de Redes Neurais Convolucionais (CNNs)

## Introdução

O câncer de pele é o tipo de câncer mais comum no Brasil e no mundo. Dentre os diferentes tipos, o melanoma é o mais perigoso devido à sua capacidade de se espalhar rapidamente para outros órgãos (metástase), tornando o diagnóstico precoce crucial para o aumento das chances de cura. A identificação visual de sinais na pele com características suspeitas (tamanho, formato, cor irregulares) é um passo importante para o diagnóstico.

Este projeto tem como objetivo desenvolver uma Rede Neural Convolucional (CNN) capaz de realizar a classificação binária de imagens de sinais de pele, categorizando-os como benignos (0) ou malignos (1). A classificação automatizada de imagens usando técnicas de aprendizado profundo como CNNs pode auxiliar dermatologistas no processo de triagem e identificação de lesões suspeitas, complementando o exame visual e potencialmente levando a diagnósticos mais rápidos e precisos.

## Dataset

O dataset utilizado neste projeto é o **Melanoma Skin Cancer Dataset of 10000 Images**, disponível no Kaggle. Este dataset é derivado do **ISIC ARCHIVE (The International Skin Imaging Collaboration)**, que é um grande arquivo público de imagens de pele com o objetivo de servir como recurso para ensino, pesquisa e desenvolvimento e teste de algoritmos de inteligência artificial de diagnóstico.

Conforme o nome sugere, o dataset contém aproximadamente 10000 imagens de sinais de pele, que são utilizadas para treinar e avaliar o modelo de classificação.

## Pré-processamento

As imagens do dataset foram pré-processadas para prepará-las para o treinamento da CNN. O pré-processamento incluiu:

*   **Normalização:** As imagens foram redimensionadas e os valores dos pixels foram reescalados para o intervalo [0, 1] dividindo-os por 255. Isso é uma prática comum em redes neurais para ajudar na convergência.
*   **Divisão em conjuntos:** O dataset foi dividido em conjuntos de treinamento, validação e teste.
*   **Data Augmentation (Aumento de Dados):** Para o conjunto de treinamento, técnicas de aumento de dados foram aplicadas utilizando `ImageDataGenerator` do TensorFlow/Keras. Isso incluiu rotações, deslocamentos de largura e altura, zoom, inversões horizontais e verticais, e ajustes de brilho. O aumento de dados ajuda a variar o conjunto de treinamento, tornando o modelo mais robusto e generalizável, reduzindo o overfitting. O conjunto de validação e teste não tiveram aumento de dados, apenas normalização.
*   **Redimensionamento:** Todas as imagens foram redimensionadas para o tamanho de 224x224 pixels, que é o tamanho de entrada esperado pela arquitetura da CNN.
*   **Geração de Batches:** `ImageDataGenerator.flow_from_directory` foi utilizado para criar geradores de dados para os conjuntos de treinamento, validação e teste. Esses geradores leem as imagens diretamente dos diretórios, aplicam as transformações e retornam batches de imagens e seus rótulos, facilitando o treinamento eficiente do modelo. O tamanho do batch utilizado foi de 32.

## Arquitetura da CNN

O modelo de Rede Neural Convolucional personalizado utilizado neste projeto segue uma arquitetura `Sequential` e foi projetado para a tarefa de classificação binária. A estrutura da rede é composta pelas seguintes camadas:

*   **Camadas Convolucionais (`Conv2D`):**
    *   Três camadas convolucionais são utilizadas com filtros progressivamente maiores: 32, 64 e 128 filtros, respectivamente.
    *   Todas as camadas convolucionais utilizam um tamanho de kernel de 3x3.
    *   A função de ativação 'relu' (Rectified Linear Unit) é aplicada em todas as camadas convolucionais para introduzir não-linearidade.
    *   A primeira camada convolucional define a `input_shape` esperada para as imagens: (224, 224, 3), indicando imagens coloridas (RGB) de 224x224 pixels.

*   **Camadas de Pooling (`MaxPooling2D`):**
    *   Cada camada convolucional é seguida por uma camada de MaxPooling com tamanho de pool de 2x2.
    *   O MaxPooling reduz as dimensões espaciais dos mapas de características, ajudando a diminuir o número de parâmetros e computação, além de tornar o modelo mais robusto a pequenas variações na posição das características.

*   **Camada de Achatamento (`Flatten`):**
    *   Após as camadas convolucionais e de pooling, uma camada `Flatten` é utilizada para transformar os mapas de características 3D resultantes em um vetor 1D. Isso é necessário para conectar as camadas convolucionais às camadas densas subsequentes.

*   **Camadas Densas (`Dense`):**
    *   Uma camada oculta densa com 128 neurônios e função de ativação 'relu'.
    *   Uma camada de saída densa com 1 neurônio. Esta camada utiliza a função de ativação 'sigmoid', que é apropriada para problemas de classificação binária, pois a saída representa a probabilidade da imagem pertencer à classe positiva (maligno).

*   **Camada de Dropout (`Dropout`):**
    *   Uma camada de Dropout com taxa de 0.5 (50%) é aplicada após a camada densa oculta.
    *   O Dropout desativa aleatoriamente uma fração dos neurônios durante o treinamento, o que ajuda a prevenir o overfitting e melhora a capacidade de generalização do modelo.

A arquitetura foi projetada para capturar padrões hierárquicos nas imagens, começando com filtros menores para detectar características de baixo nível e progredindo para filtros maiores para características mais complexas, enquanto reduz as dimensões espaciais através do pooling.

## Treinamento e Avaliação

### Compilação do Modelo

O modelo customizado foi compilado utilizando os seguintes parâmetros:

*   **Otimizador:** `adam` - Um otimizador adaptativo popular e eficaz para uma ampla gama de problemas de aprendizado de máquina.
*   **Função de Perda:** `binary_crossentropy` - A função de perda apropriada para problemas de classificação binária. Ela mede o quão bem o modelo está prevendo a probabilidade da classe positiva.
*   **Métricas:** O modelo foi avaliado utilizando 'accuracy' (acurácia), `Precision` (precisão) e `Recall` (revocação). Essas métricas fornecem uma visão mais completa do desempenho do modelo, especialmente em datasets desbalanceados ou onde o custo de falsos positivos e falsos negativos difere.

### Treinamento

O modelo foi treinado utilizando o método `fit` com os geradores de dados de treinamento (`train_generator`) e validação (`val_generator`). O treinamento foi realizado por 10 épocas.

### Histórico de Treinamento

O histórico de treinamento registra as métricas de perda e desempenho (acurácia, precisão, revocação) para os conjuntos de treinamento e validação em cada época. O gráfico do histórico de treinamento mostra a evolução dessas métricas ao longo das épocas.

*   As curvas de perda (loss) para treino e validação indicam como o erro do modelo diminui durante o treinamento. Idealmente, ambas devem diminuir e se manter próximas. Uma grande diferença entre elas pode indicar overfitting (o modelo está aprendendo demais os dados de treino e não generaliza bem para dados novos).
*   As curvas de acurácia, precisão e revocação para treino e validação mostram como o desempenho do modelo melhora em termos de classificação correta. Uma lacuna crescente entre as métricas de treino e validação também pode ser um sinal de overfitting.

Observando o gráfico do histórico de treinamento, é possível analisar se o modelo está aprendendo de forma adequada, se está ocorrendo overfitting ou underfitting, e se o número de épocas foi suficiente ou excessivo.

### Avaliação no Conjunto de Teste

Após o treinamento, o modelo foi avaliado no conjunto de teste (`test_generator`) para estimar seu desempenho em dados não vistos. A avaliação foi realizada utilizando o `classification_report` do scikit-learn.

Para a tarefa de diagnóstico de melanoma, a **Revocação (Recall)** é uma métrica crucial. Um alto recall significa que o modelo é bom em identificar a maioria dos casos positivos reais (melanomas malignos), minimizando os falsos negativos (melanomas que não são detectados). Falsos negativos podem ter consequências graves em aplicações médicas. Por outro lado, a **Precisão (Precision)** indica a proporção de positivos verdadeiros entre todos os casos classificados como positivos.

Os resultados da avaliação no conjunto de teste são apresentados no `classification_report`, incluindo precisão, revocação e f1-score para cada classe (benigno e maligno), além da acurácia geral.

*   **Acurácia (Accuracy):** A proporção de previsões corretas sobre o total de previsões. No contexto médico, embora útil, não é a única métrica a ser considerada, especialmente em datasets com classes desbalanceadas.
*   **Precisão (Precision):** Para a classe maligno, indica a proporção de sinais preditos como malignos que são realmente malignos.
*   **Revocação (Recall):** Para a classe maligno, indica a proporção de sinais malignos reais que foram corretamente identificados pelo modelo. Uma alta revocação é desejada para minimizar falsos negativos.

No caso deste projeto, o limiar de classificação foi ajustado para 0.4 (em vez do padrão 0.5) para a saída sigmoide. Isso foi feito com o objetivo de aumentar a revocação para a classe maligno, reduzindo assim o número de falsos negativos, o que é clinicamente preferível (é melhor classificar um sinal benigno como maligno e investigá-lo, do que não detectar um melanoma).

### Matriz de Confusão

A matriz de confusão fornece uma visualização detalhada do desempenho do classificador, mostrando o número de previsões corretas e incorretas para cada classe.

A matriz de confusão tem os seguintes quadrantes:

*   **Verdadeiros Positivos (True Positives - TP):** Casos malignos que foram corretamente preditos como malignos.
*   **Verdadeiros Negativos (True Negatives - TN):** Casos benignos que foram corretamente preditos como benignos.
*   **Falsos Positivos (False Positives - FP):** Casos benignos que foram incorretamente preditos como malignos.
*   **Falsos Negativos (False Negatives - FN):** Casos malignos que foram incorretamente preditos como benignos.

A precisão e a revocação podem ser calculadas a partir da matriz de confusão:

*   Precision = TP / (TP + FP)
*   Recall = TP / (TP + FN)

A análise da matriz de confusão, juntamente com o classification report, ajuda a entender onde o modelo está acertando e errando, e a relação entre os falsos positivos e falsos negativos, que é crucial para a avaliação em um contexto médico.

## Análise dos Erros

Uma análise detalhada dos erros cometidos pelo modelo no conjunto de teste foi realizada, focando nos Falsos Positivos e Falsos Negativos. A visualização das imagens classificadas incorretamente (como demonstrado no notebook) fornece insights importantes sobre as limitações do modelo.

### Falsos Positivos (Predito: Maligno, Real: Benigno)

As imagens classificadas incorretamente como malignas (falsos positivos) frequentemente correspondem a sinais benignos que, visualmente, apresentam características que levantam suspeita mesmo para um olho treinado, como bordas irregulares, diâmetro considerável e variações de cor. Em um cenário clínico, é provável que tais lesões fossem removidas por precaução para análise histopatológica, onde seria confirmado seu caráter benigno.

**Potenciais Causas:** A ambiguidade visual dessas lesões e a semelhança de certas características morfológicas entre lesões benignas atípicas e melanomas iniciais podem levar o modelo a classificá-las como suspeitas. O modelo pode estar supervalorizando certas características visuais que são comuns tanto em benignos suspeitos quanto em malignos.

**Implicações Clínicas:** Falsos positivos, embora causem ansiedade e levem a procedimentos desnecessários (biópsias), são geralmente preferíveis aos falsos negativos no diagnóstico de melanoma, pois não resultam na falha em detectar um câncer.

### Falsos Negativos (Predito: Benigno, Real: Maligno)

As imagens classificadas incorretamente como benignas (falsos negativos) são de maior preocupação clínica, pois representam melanomas que não foram detectados pelo modelo. Observou-se que alguns desses falsos negativos correspondem a lesões malignas que não possuem bordas tão claramente delimitadas ou que se assemelham mais a lesões benignas em suas características superficiais.

**Potenciais Causas:** Melanomas em estágios muito iniciais ou com morfologia atípica podem ser difíceis de distinguir visualmente de lesões benignas. O modelo pode ter dificuldade em identificar os padrões sutis que diferenciam essas lesões, especialmente se o dataset de treinamento não contiver exemplos suficientes e variados desses casos desafiadores. A qualidade da imagem e a presença de artefatos também podem influenciar a detecção.

**Implicações Clínicas:** Falsos negativos são criticamente importantes em aplicações médicas, pois um melanoma não detectado pode se desenvolver e metastatizar, diminuindo drasticamente as chances de tratamento bem-sucedido. A minimização dos falsos negativos foi a principal razão para ajustar o limiar de classificação, priorizando o recall (revocação).

A análise dos erros reforça a complexidade do diagnóstico de melanoma e a importância de considerar as características visuais das lesões ao interpretar as previsões do modelo. Embora o modelo demonstre boa acurácia geral, a natureza dos erros destaca áreas onde a interpretação clínica humana continua sendo essencial.

## Instalação e Uso

Para executar este projeto, é necessário ter o ambiente Python configurado com as seguintes dependências:

*   TensorFlow
*   Keras
*   scikit-learn
*   matplotlib
*   numpy
*   mlxtend

Você pode instalar a maioria dessas dependências utilizando pip:

```bash
pip install tensorflow keras scikit-learn matplotlib numpy mlxtend
```

**Observação:** Dependendo do seu ambiente (local ou Google Colab, como utilizado neste notebook), pode ser necessário instalar versões específicas ou ter configurações adicionais (como suporte a GPU para TensorFlow).

### Estrutura do Dataset

O código espera que o dataset esteja organizado em diretórios de treinamento (`train`) e teste (`test`), e dentro desses, subdiretórios para cada classe (`benign` e `malignant`). Por exemplo:

```
/caminho/para/seu/dataset/
├── train/
│   ├── benign/
│   └── malignant/
└── test/
    ├── benign/
    └── malignant/
```

No contexto deste notebook, os dados foram copiados para os diretórios `/content/train` e `/content/test`.

### Fluxo de Execução

O código pode ser executado sequencialmente em um ambiente Python com as bibliotecas instaladas. O fluxo geral é o seguinte:

1.  **Pré-processamento:** Carrega as imagens utilizando `ImageDataGenerator`, aplicando normalização, aumento de dados (para treino) e dividindo em conjuntos de treino, validação e teste.
2.  **Construção/Carga do Modelo:** Define a arquitetura da CNN personalizada (ou carrega um modelo pré-treinado, se disponível).
3.  **Compilação do Modelo:** Configura o otimizador, função de perda e métricas para o treinamento.
4.  **Treinamento do Modelo:** Treina a CNN utilizando os geradores de dados de treino e validação.
5.  **Avaliação do Modelo:** Avalia o desempenho do modelo no conjunto de teste utilizando métricas como acurácia, precisão, revocação e matriz de confusão.
6.  **Análise de Erros:** Visualiza e analisa as imagens classificadas incorretamente para entender as limitações do modelo.

Ao rodar o código, siga a ordem das células no notebook para garantir que as etapas sejam executadas corretamente.

## Conclusão

Este projeto demonstrou a viabilidade de utilizar uma Rede Neural Convolucional personalizada para a classificação binária de sinais de pele em benignos e malignos para auxiliar no diagnóstico de melanoma.

O modelo customizado alcançou uma acurácia geral de 89% no conjunto de teste. Mais importante para o contexto clínico de detecção de melanoma, onde a minimização de falsos negativos é crucial, o modelo obteve métricas de precisão e revocação para a classe maligno próximas a 88% e 90%, respectivamente (após ajuste do limiar de classificação para 0.4 para priorizar a revocação). A análise da matriz de confusão corroborou esses resultados, mostrando um bom equilíbrio entre verdadeiros positivos e negativos, com uma taxa de falsos negativos relativamente baixa, que é clinicamente desejável.

A análise visual dos erros revelou que os falsos positivos muitas vezes correspondem a lesões benignas com características suspeitas, enquanto os falsos negativos podem estar associados a melanomas menos definidos ou em estágios iniciais. Isso ressalta a complexidade inerente ao diagnóstico visual e a importância da interpretação humana em conjunto com as ferramentas de IA.

### Melhorias Futuras

Este projeto serve como um ponto de partida e há diversas áreas para futuras melhorias:

*   **Arquiteturas Mais Avançadas:** Explorar o uso de modelos pré-treinados em grandes datasets de imagens (como ImageNet) através de Transfer Learning (por exemplo, utilizando ResNet, Inception, EfficientNet). Isso pode capturar características visuais mais ricas e complexas e potencialmente levar a um desempenho superior, especialmente com datasets de tamanho moderado.
*   **Aumento de Dados e Técnicas de Balanceamento:** Experimentar com técnicas de aumento de dados mais sofisticadas ou específicas para imagens médicas. Além disso, investigar técnicas para lidar com o possível desbalanceamento de classes no dataset (embora o dataset atual pareça razoavelmente balanceado), como oversampling de classes minoritárias (SMOTE) ou técnicas de ponderação de classes durante o treinamento.
*   **Datasets Maiores e Mais Diversos:** Treinar o modelo em datasets maiores e mais diversos (incluindo imagens de diferentes fontes, etnias de pele, e variações na qualidade da imagem) pode melhorar a robustez e generalização do modelo.
*   **Integração em Aplicação:** Desenvolver uma interface gráfica ou aplicação web simples onde usuários possam carregar uma imagem de sinal de pele e obter a predição do modelo.
*   **Interpretabilidade do Modelo:** Investigar técnicas de interpretabilidade (como Grad-CAM) para visualizar quais partes da imagem a CNN está utilizando para fazer suas predições. Isso pode ajudar a construir confiança no modelo e fornecer insights clínicos.
*   **Modelos de Segmentação ou Detecção:** Em vez de apenas classificar a imagem inteira, desenvolver modelos que segmentem a área do sinal na imagem ou detectem múltiplos sinais, o que pode ser mais útil em cenários reais com várias lesões.

Em suma, o modelo customizado demonstrou um desempenho promissor para a classificação de sinais de pele, com um foco apropriado na minimização de falsos negativos. As melhorias futuras propostas visam aumentar ainda mais a acurácia, robustez e utilidade clínica do sistema.
