# On Large-Batch Training for Deep Learning: Investigating the Generalization Gap and the Geometry of Local Minima 🧠📊

Este repositório contém o projeto de pesquisa, reprodução computacional e extensão prática desenvolvido para a disciplina de **Tópicos Avançados em Aprendizado de Máquina (CK0255)** da Universidade Federal do Ceará (UFC). 

O trabalho investiga o fenômeno do **Generalization Gap** (lacuna de generalização) em redes neurais profundas, tomando como base o artigo científico seminal de **Keskar et al. (ICLR, 2017)**. O ecossistema prático valida as hipóteses dos autores tanto em cenários clássicos de classificação quanto em tarefas complexas de detecção de objetos no setor de transporte e tráfego.

---

## 🔬 Fundamentação Teórica

O uso de grandes lotes (*Large-Batch*) é uma estratégia crucial para explorar o paralelismo massivo de GPUs e TPUs modernas, reduzindo drasticamente o tempo de treino. No entanto, modelos treinados com grandes lotes tendem a apresentar uma degradação na performance ao operar sobre dados inéditos (conjunto de teste/validação). 

Este projeto analisa as duas frentes que explicam esse comportamento:

1. **A Geometria dos Mínimos Locais (Sharp vs. Flat Minima):**
   * **Small-Batch (SB):** A natureza estocástica do gradiente introduz um ruído que atua como um regularizador natural, empurrando o otimizador para fora de vales estreitos e confinando o modelo em **mínimos planos (*Flat Minima*)**. Esses vales toleram pequenas variações estruturais entre os dados de treino e teste.
   * **Large-Batch (LB):** A aproximação do gradiente é estável e determinística, fazendo com que o algoritmo convirja para os mínimos agudos mais próximos (**mínimos agudos (*Sharp Minima*)**). Nestes vales estreitos, qualquer discrepância sutil na superfície de perda resulta em uma perda acentuada de generalização.
2. **Efeito dos Otimizadores:** Análise do comportamento exploratório do algoritmo **SGD Puro** em contraposição ao comportamento de convergência acelerada de algoritmos baseados em momentos adaptativos, como o **Adam**.

---

## 📂 Arquitetura do Repositório e Código

O repositório está estruturado de forma modular, separando a pesquisa conceitual, os scripts de treinamento e os dados analíticos coletados:

* **`Proposta de Projeto.pdf`**: Planejamento bibliográfico, delimitação do escopo e hipóteses iniciais de modelagem.
* **`Relatorio Final - ML.pdf`**: Relatório científico completo detalhando as deduções matemáticas, metodologia experimental e discussões críticas sobre os resultados de agudeza (*sharpness*).
* **`Slides Apresentação.pdf`**: Apresentação visual utilizada na defesa do projeto, sintetizando os gráficos de evolução de perda.
* **`src/vgg.ipynb`**: Notebook contendo a pipeline de reprodução do artigo base, focando em arquiteturas convolucionais clássicas (**VGG**) e análise comparativa de otimização (SGD vs. Adam).
* **`src/yolo.ipynb`**: pipeline de extensão do projeto para detecção de objetos utilizando a arquitetura **YOLO** calibrada sobre um dataset privado de trânsito para identificação de veículos.
* **`src/batch16/` e `src/batch128/`**: Diretórios que concentram os outputs nativos dos experimentos da YOLO, contendo:
    * `results.csv` e `results.png`: Histórico detalhado de perdas por época.
    * `confusion_matrix.png` e `confusion_matrix_normalized.png`: Matrizes de confusão para validação de falso-positivos.
    * `BoxF1_curve.png`, `BoxPR_curve.png`, `BoxP_curve.png`, `BoxR_curve.png`: Curvas de Precisão, Recall e F1-Score.
    * Imagens de amostragem de treino e predição (`train_batch*.jpg`, `val_batch*_pred.jpg`).

---

## 📈 Resultados Experimentais Destacados

Os testes comparativos na arquitetura YOLO operando com lote pequeno (16) versus lote grande (128) sob as mesmas condições de treino evidenciaram o comportamento previsto por Keskar et al.:

| Métrica Analisada | Small-Batch (Lote 16) | Large-Batch (Lote 128) | Impacto Geométrico |
| :--- | :---: | :---: | :--- |
| **Acurácia de Treino ($mAP_{50}$)** | $0.672$ | $0.672$ | Ambos atingem convergência perfeita |
| **Acurácia de Validação ($mAP_{50}$)** | $0.603$ | $0.590$ | **Generalization Gap** evidenciado no lote maior |
| **Sharpness Média Estimada** | $0.17\%$ | $0.46\%$ | Lote grande retido em mínimos agudos (*Sharp*) |

As curvas de aprendizado extraídas do arquivo `results.png` indicam que o comportamento de *Box Loss* do Lote 128 sofre uma estagnação precoce na validação devido à hipersensibilidade do modelo às variações locais do cenário de tráfego.

---

## 🔧 Como Executar e Reproduzir

### Pré-requisitos
Certifique-se de possuir um ambiente Python 3.10+ configurado com suporte a aceleração por GPU (CUDA) e as seguintes bibliotecas fundamentais:
```bash
pip install ultralytics torch numpy pandas matplotlib jupyter
```
### Execução dos Experimentos
1. Para analisar e rodar a classificação clássica com VGG:
```bash
jupyter notebook src/vgg.ipynb
```

2. Para rodar a pipeline de detecção e análise de hiperparâmetros (Lote 16 vs 128) com a YOLO:
```bash
jupyter notebook src/yolo.ipynb
