#!/bin/bash

# Navega para o diretório do script
cd "$(dirname "$0")"

# Ativa o ambiente virtual (se existir)
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Executa o Streamlit em background na porta 8503 (sem abrir navegador automaticamente)
streamlit run app.py --server.port=8503 --server.headless=true &
STREAMLIT_PID=$!

# Aguarda alguns segundos para o servidor iniciar
sleep 3

# Abre o navegador no endereço do Streamlit
open http://localhost:8503

# Aguarda o processo do Streamlit terminar
wait $STREAMLIT_PID

