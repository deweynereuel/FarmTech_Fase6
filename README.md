# FarmTech Solutions: Visão Computacional com YOLOv5

Projeto da disciplina de Redes Neurais (PBL Fase 6), FIAP. Sistema de detecção de **cadeiras** e **garrafas**, desenvolvido para demonstrar a um cliente fictício da FarmTech Solutions como a visão computacional funciona na prática e como diferentes abordagens se comparam.

## 👥 Grupo Farm Tech

- **Jeliel Cardoso** — [LinkedIn](https://www.linkedin.com/in/jelielcardoso/)
- **Denis Paulo Dias da Silva** — [LinkedIn](https://www.linkedin.com/in/denispaulodiassilva/)
- **Deweyne Reuel** — [LinkedIn](https://www.linkedin.com/in/deweyne-reuel-0695522a8/)
- **Matheus Nascimento** — [LinkedIn](https://www.linkedin.com/in/mathnascimento/)


## 👥 Professores

- **Tutor(a):** [Sabrina Otoni](https://www.linkedin.com/in/sabrina-otoni-22525519b/)
- **Coordenador(a):** [André Godoi](https://www.linkedin.com/in/andregodoichiovato/)
## 🧭 Entregas do Projeto

| Entregas| Local |
|---|---|
| **Passo a passo completo, código executado, resultados e conclusões (Entregas 1 e 2)** | 👉 [`DeweyneReuel_rm573755_pbl_fase6.ipynb`](./DeweyneReuel_rm573755_pbl_fase6.ipynb) · [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IY_VyBb9cX4Oh0IEY4ZG5y5CJGfDbSWd?usp=drive_link) |
| **Vídeo de demonstração** (até 5 min, YouTube não listado) | [Assistir](https://youtu.be/L8017RfiJs8) |
| **Dataset** (80 imagens + rótulos, no Google Drive) | Pasta com acesso por link, enviado no PDF de entrega pelo portal da FIAP |
| **Rótulos no formato YOLO** | [`labels/`](pbl_farmtech_fase6/Farmtech_YOLO/labels) |
| **Imagens usadas no treinamento** | [`images/`](pbl_farmtech_fase6/Farmtech_YOLO/images) |

## 🎯 Contexto e proposta

A FarmTech Solutions está ampliando seus serviços de IA para além do agronegócio, incluindo a visão computacional. O projeto tem duas entregas, ambas documentadas no mesmo notebook:

- **Entrega 1:** customização do **YOLOv5** para detectar `chair` e `bottle` em um dataset próprio de 80 imagens (40 de cada), rotuladas no [Make Sense](https://www.makesense.ai/). O modelo é treinado com 30, 60 e 90 épocas e avaliado em validação e teste.
- **Entrega 2:** comparação do YOLO customizado com duas abordagens concorrentes sobre a mesma base: a **YOLOv3 tradicional** (Darknet, pesos COCO, executada via OpenCV) e uma **CNN treinada do zero** (TensorFlow/Keras) que classifica a imagem inteira. A comparação cobre facilidade de uso, precisão, tempo de treinamento/customização e tempo de inferência.

**Toda a descrição técnica, as tabelas e as conclusões estão no notebook**; este README serve de porta de entrada.

## ▶️ Como executar

1. Abra o notebook no Google Colab e ative a GPU (*Ambiente de execução > Alterar tipo de ambiente de execução > T4 GPU*).
2. Adicione a pasta do dataset ao seu *Meu Drive* (link no PDF de entrega) com o nome `Farmtech_YOLO`. Se usar outro caminho, ajuste a variável `DRIVE_DS` na primeira célula de configuração.
3. Execute todas as células em ordem. Em um Drive sem os pesos treinados, o notebook treina os três modelos YOLO (cerca de 10 minutos no total) e baixa os arquivos da YOLOv3 (aprox. 240 MB). Se os pesos já existirem no Drive, o treino é pulado.

Os resultados podem variar um pouco entre execuções (o treinamento tem componentes aleatórios e o conjunto de teste é pequeno); os valores analisados correspondem à execução registrada no notebook.

## 📁 Estrutura do repositório

```
.
├── ir_alem_web_cam/   # A estrutura e descrição do ir além será abordada em ir_alem_web_cam/README.md
|   ├── docs/
|   ├── evidencias/
|   ├── notebooks/
|   ├── src/
|   ├── .gitignore
|   ├── README.md
|   └── requirements.txt
├── pbl_farmtech_fase6/   
|   ├── Farmtech_YOLO/
|   |   ├── images/    # Contém as imagens utilizadas no treinamento
|   |   |   ├── test/  # Imagens de test     
|   |   |   ├── train/ # Imagens de treino
|   |   |   └── val/   # Imagens de validação
|   |   └── labels/    # Contém os rótulos das imagens
|   |       ├── test/  # Rótulos das imagens de test
|   |       ├── train/ # Rótulos das imagens de treino
|   |       └── val/   # Rótulos das imagens de validação
|   ├── data.yaml
|   └── DeweyneReuel_rm573755_pbl_fase6_ipynb
├── README.md                               

```

## ⚠️ Observações

- As imagens do dataset foram obtidas da internet e são usadas **apenas para fins acadêmicos**.
- Ferramentas: Python, PyTorch, [YOLOv5 (Ultralytics)](https://github.com/ultralytics/yolov5), OpenCV (YOLOv3 Darknet), TensorFlow/Keras, Google Colab e Google Drive.
