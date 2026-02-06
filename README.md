# Sistema de Contagem de Multidão 

Sistema de contagem de pessoas em imagens usando deep learning com a arquitetura CSRNet.

## 🎯 Características

- Interface web moderna com Streamlit
- Modelo CSRNet pré-treinado para contagem de multidões
- Visualização de mapa de densidade (heatmap)
- Logo Poder360 integrada
- Fácil de usar através de interface web

## 📋 Requisitos

- Python 3.8+
- PyTorch 2.0.1
- Streamlit
- Ver `requirements.txt` para lista completa de dependências

## 🚀 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/pedroabreutech/crowd-counting-csrnet.git
cd crowd-counting-csrnet
```

2. Crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Certifique-se de que o arquivo `weights.pth` está presente no diretório raiz do projeto.

## 💻 Uso

### Interface Web (Recomendado)

Execute a aplicação Streamlit:
```bash
streamlit run app.py
```

A interface abrirá automaticamente no navegador em `http://localhost:8501`

### Linha de Comando

Para usar via terminal:
```bash
python run.py
```

**Nota:** O script `run.py` processa automaticamente o primeiro arquivo `.jpg` encontrado no diretório.

## 📁 Estrutura do Projeto

```
Sistema_Contagem/
├── app.py              # Interface web Streamlit
├── model.py            # Arquitetura CSRNet
├── run.py              # Script de linha de comando
├── weights.pth         # Pesos do modelo pré-treinado
├── poder.png           # Logo Poder360
├── requirements.txt    # Dependências do projeto
└── README.md          # Este arquivo
```

## 🛠️ Tecnologias Utilizadas

- **PyTorch** - Framework de deep learning
- **Streamlit** - Interface web
- **PIL/Pillow** - Processamento de imagens
- **NumPy** - Operações numéricas
- **Matplotlib** - Visualização de dados

## 📊 Como Funciona

1. O modelo CSRNet carrega os pesos pré-treinados (`weights.pth`)
2. A imagem é pré-processada e normalizada
3. O modelo gera um mapa de densidade indicando onde há pessoas
4. A contagem total é calculada somando os valores do mapa de densidade
5. Um heatmap visual é gerado para mostrar a distribuição das pessoas

## 📝 Notas

- O arquivo `weights.pth` é necessário para o funcionamento do sistema
- O modelo foi treinado para contar pessoas em imagens de multidões
- Funciona melhor com imagens aéreas ou de grandes aglomerações

## 👤 Autor

Desenvolvido por Pedro 

## 📄 Licença

Este projeto é de uso privado 

