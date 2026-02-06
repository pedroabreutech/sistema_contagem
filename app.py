import streamlit as st
import torch
import os
from model import CSRNet
from PIL import Image
import numpy as np
from torchvision import transforms
from matplotlib import pyplot as plt
import base64
from datetime import datetime
import json

# -----------------------------
# CONFIGURAÇÃO DA PÁGINA
# -----------------------------
st.set_page_config(
    page_title="Crowd Counting System",
    page_icon="📸",
    layout="wide"
)

# -----------------------------
# FUNÇÃO PARA CARREGAR LOGO
# -----------------------------
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "poder.png")

# -----------------------------
# TÍTULO PERSONALIZADO COM LOGO
# -----------------------------
if os.path.exists(LOGO_PATH):
    logo_base64 = get_base64_image(LOGO_PATH)
    title_html = f"""
    <h1 style='text-align: center; color: #0066cc; font-size: 42px; font-weight: 700; display: flex; align-items: center; justify-content: center; gap: 15px;'>
        Crowd Counting System <img src="data:image/png;base64,{logo_base64}" style="height: 50px; width: auto; vertical-align: middle;">
    </h1>
    """
else:
    title_html = """
    <h1 style='text-align: center; color: #0066cc; font-size: 42px; font-weight: 700;'>
        Crowd Counting System
    </h1>
    """

st.markdown(
    f"""
    {title_html}
    <p style='text-align: center; margin-top: -10px; color: #444; font-size: 18px;'>
        
    </p>
    <br>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """<hr style="border:1px solid #e6e6e6; margin-top:-20px; margin-bottom:30px;">""",
    unsafe_allow_html=True
)

# -----------------------------
# TRANSFORMAÇÃO PARA O MODELO
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# FUNÇÃO PARA CARREGAR O MODELO
# -----------------------------
@st.cache_resource
def load_model():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, "weights.pth")
        
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Erro: Arquivo de pesos do modelo não encontrado em {MODEL_PATH}")
            st.stop()
            return None

        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        model = CSRNet()
        model.load_state_dict(checkpoint)
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Erro ao carregar o modelo: {str(e)}")
        st.stop()
        return None

# Carregar modelo apenas quando necessário (lazy loading)
model = None

# -----------------------------
# FUNÇÕES DE CÁLCULO DE MÉTRICAS
# -----------------------------
def calculate_accuracy_metrics(predicted_count, true_count):
    """Calcula métricas de acurácia quando há contagem real"""
    error_absolute = abs(predicted_count - true_count)
    error_percentage = (error_absolute / true_count * 100) if true_count > 0 else 0
    accuracy = max(0, 100 - error_percentage)
    
    return {
        'error_absolute': error_absolute,
        'error_percentage': error_percentage,
        'accuracy': accuracy
    }

def calculate_confidence_metrics(density_map, predicted_count):
    """Calcula métricas de confiança baseadas no mapa de densidade - Versão Melhorada"""
    # Estatísticas básicas
    variance = np.var(density_map)
    std_dev = np.std(density_map)
    max_density = np.max(density_map)
    mean_density = np.mean(density_map)
    median_density = np.median(density_map)
    
    # Remover zeros para cálculos mais precisos
    non_zero_values = density_map[density_map > 0]
    has_valid_values = len(non_zero_values) > 0
    
    # Fator 1: Coeficiente de Variação (usando escala logarítmica suavizada)
    cv = (std_dev / mean_density * 100) if mean_density > 0 else 1000
    # Usar função logarítmica para suavizar cv alto
    cv_score = 100 / (1 + np.log10(1 + cv / 10))  # Converte cv alto em score baixo de forma suave
    cv_score = max(0, min(100, cv_score))
    
    # Fator 2: Razão Mediana/Média (medida de simetria - quanto mais próximo de 1, melhor)
    median_mean_ratio = (median_density / mean_density) if mean_density > 0 else 0
    symmetry_score = min(100, median_mean_ratio * 100)  # Ideal: mediana = média
    
    # Fator 3: Densidade média absoluta (mapas com densidade muito baixa são menos confiáveis)
    # Normalizar baseado em densidade típica (ajustar conforme necessário)
    density_threshold = 0.001  # Densidade mínima esperada
    density_score = min(100, (mean_density / density_threshold) * 20)  # Score baseado na densidade
    density_score = max(20, density_score)  # Mínimo de 20% mesmo para densidades muito baixas
    
    # Fator 4: Concentração vs Espalhamento (usando percentis)
    # Mapas muito concentrados ou muito dispersos podem indicar menor confiança
    q25 = np.percentile(density_map, 25)
    q75 = np.percentile(density_map, 75)
    iqr = q75 - q25
    concentration_ratio = (iqr / mean_density * 100) if mean_density > 0 else 100
    concentration_score = 100 / (1 + concentration_ratio / 50)  # Score baseado na concentração
    concentration_score = max(0, min(100, concentration_score))
    
    # Fator 5: Razão Max/Mean (outliers extremos reduzem confiança)
    max_mean_ratio = (max_density / mean_density) if mean_density > 0 else 1000
    # Valores muito altos indicam outliers extremos
    outlier_score = 100 / (1 + np.log10(1 + max_mean_ratio / 10))
    outlier_score = max(0, min(100, outlier_score))
    
    # Fator 6: Porcentagem de pixels não-zero (cobertura da detecção)
    if has_valid_values:
        coverage = (len(non_zero_values) / density_map.size) * 100
        coverage_score = min(100, coverage * 1.5)  # Score baseado na cobertura
    else:
        coverage = 0
        coverage_score = 10  # Score muito baixo se não há detecções
    
    # Pesos para cada fator (ajustáveis conforme necessário)
    weights = {
        'cv': 0.25,           # Coeficiente de variação é importante
        'symmetry': 0.15,     # Simetria indica distribuição equilibrada
        'density': 0.15,      # Densidade absoluta é relevante
        'concentration': 0.15, # Concentração adequada
        'outliers': 0.15,     # Outliers reduzem confiança
        'coverage': 0.15      # Cobertura da detecção
    }
    
    # Score de confiança combinado (média ponderada)
    confidence_score = (
        cv_score * weights['cv'] +
        symmetry_score * weights['symmetry'] +
        density_score * weights['density'] +
        concentration_score * weights['concentration'] +
        outlier_score * weights['outliers'] +
        coverage_score * weights['coverage']
    )
    
    # Garantir que está no range [0, 100]
    confidence_score = max(0, min(100, confidence_score))
    
    # Margem de erro adaptativa baseada no score de confiança
    # Score alto = margem menor, Score baixo = margem maior
    base_error_margin = 12  # Margem base padrão
    # Ajustar margem inversamente ao score: score 100% = 8%, score 0% = 20%
    estimated_error_margin = base_error_margin + ((100 - confidence_score) / 100) * 8
    estimated_error_margin = max(8, min(20, estimated_error_margin))  # Limitar entre 8% e 20%
    
    # Calcular intervalo de confiança
    lower_bound = int(predicted_count * (1 - estimated_error_margin / 100))
    upper_bound = int(predicted_count * (1 + estimated_error_margin / 100))
    lower_bound = max(0, lower_bound)  # Não pode ser negativo
    
    return {
        'variance': variance,
        'std_dev': std_dev,
        'max_density': max_density,
        'mean_density': mean_density,
        'median_density': median_density,
        'coefficient_variation': cv,
        'confidence_score': confidence_score,
        'estimated_error_margin': estimated_error_margin,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'component_scores': {
            'cv_score': cv_score,
            'symmetry_score': symmetry_score,
            'density_score': density_score,
            'concentration_score': concentration_score,
            'outlier_score': outlier_score,
            'coverage_score': coverage_score
        },
        'coverage_percentage': coverage,
        'iqr': iqr,
        'max_mean_ratio': max_mean_ratio
    }

# -----------------------------
# UPLOAD DA IMAGEM
# -----------------------------
st.subheader("📤 Envie uma imagem para análise")

st.info("💡 **Como usar:** Selecione uma imagem de multidão ou aglomeração usando o seletor abaixo. O sistema irá analisar a imagem e contar automaticamente o número de pessoas.")

uploaded_file = st.file_uploader(
    "Selecione uma imagem",
    type=["jpg", "jpeg", "png"],
    help="Envie fotos aéreas, de multidões ou grandes aglomerações.",
    label_visibility="visible"
)

if not uploaded_file:
    st.markdown("---")
    st.markdown("### 📋 Instruções")
    st.markdown("""
    1. **Selecione uma imagem** usando o seletor acima
    2. Aguarde o processamento automático
    3. Visualize os resultados da contagem
    4. (Opcional) Informe a contagem real para calcular a acurácia
    """)

# -----------------------------
# PROCESSAMENTO DA IMAGEM
# -----------------------------
if uploaded_file:
    # Carregar modelo apenas quando necessário
    if model is None:
        with st.spinner("🔄 Carregando modelo de contagem..."):
            model = load_model()
    
    if model is None:
        st.error("❌ Não foi possível carregar o modelo. Verifique se o arquivo weights.pth existe.")
        st.stop()
    
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagem enviada", use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Preprocessamento
    with st.spinner("🔄 Processando imagem e contando pessoas..."):
        img_tensor = transform(image)
        output = model(img_tensor.unsqueeze(0))
        count = int(output.detach().cpu().sum().numpy())
        density_map = output.detach().cpu().numpy()[0][0]

    # -----------------------------
    # CAMPO PARA CONTAGEM REAL (OPCIONAL)
    # -----------------------------
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Resultados da Análise")
    
    with col2:
        has_ground_truth = st.checkbox("Tenho a contagem real", help="Marque se você conhece a contagem real de pessoas na imagem")
    
    if has_ground_truth:
        true_count = st.number_input(
            "Contagem Real (número de pessoas)",
            min_value=0,
            value=count,
            step=1,
            help="Informe o número real de pessoas na imagem para calcular a acurácia"
        )
    
    # -----------------------------
    # CÁLCULO DAS MÉTRICAS
    # -----------------------------
    confidence_metrics = calculate_confidence_metrics(density_map, count)
    
    if has_ground_truth and 'true_count' in locals():
        accuracy_metrics = calculate_accuracy_metrics(count, true_count)
    else:
        accuracy_metrics = None

    # -----------------------------
    # CARD DO RESULTADO PRINCIPAL
    # -----------------------------
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            f"""
            <div style="padding: 20px; border-radius: 12px; background: #f0f7ff; border: 1px solid #cce0ff;">
                <h2 style="color:#004c99; margin:0; font-size: 24px;">
                    📊 Estimativa: <b>{count}</b> pessoas
                </h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        confidence_color = "#28a745" if confidence_metrics['confidence_score'] >= 70 else "#ffc107" if confidence_metrics['confidence_score'] >= 50 else "#dc3545"
        st.markdown(
            f"""
            <div style="padding: 20px; border-radius: 12px; background: #f8f9fa; border: 1px solid #dee2e6;">
                <h2 style="color:{confidence_color}; margin:0; font-size: 24px;">
                    🎯 Confiança: <b>{confidence_metrics['confidence_score']:.1f}%</b>
                </h2>
                <p style="color:#666; margin:5px 0 0 0; font-size: 14px;">
                    Intervalo: {confidence_metrics['lower_bound']} - {confidence_metrics['upper_bound']}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------
    # RELATÓRIO DETALHADO
    # -----------------------------
    st.subheader("📋 Relatório Detalhado")
    
    report_tabs = st.tabs(["📈 Métricas de Confiança", "✅ Análise de Acurácia", "📄 Relatório Completo"])
    
    with report_tabs[0]:
        st.markdown("### Métricas Baseadas no Modelo")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Score de Confiança",
                f"{confidence_metrics['confidence_score']:.1f}%",
                help="Indica o nível de confiança do modelo baseado na consistência do mapa de densidade"
            )
        
        with col2:
            st.metric(
                "Margem de Erro Estimada",
                f"±{confidence_metrics['estimated_error_margin']}%",
                help="Margem de erro estimada com base em estatísticas do modelo"
            )
        
        with col3:
            st.metric(
                "Intervalo Estimado",
                f"{confidence_metrics['lower_bound']} - {confidence_metrics['upper_bound']}",
                help="Faixa provável da contagem real"
            )
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Estatísticas do Mapa de Densidade:**")
            st.write(f"- Variância: {confidence_metrics['variance']:.4f}")
            st.write(f"- Desvio Padrão: {confidence_metrics['std_dev']:.4f}")
            st.write(f"- Densidade Máxima: {confidence_metrics['max_density']:.4f}")
            st.write(f"- Densidade Média: {confidence_metrics['mean_density']:.4f}")
            st.write(f"- Densidade Mediana: {confidence_metrics['median_density']:.4f}")
            st.write(f"- Coeficiente de Variação: {confidence_metrics['coefficient_variation']:.2f}%")
            st.write(f"- Intervalo Interquartil (IQR): {confidence_metrics['iqr']:.4f}")
            st.write(f"- Cobertura de Detecção: {confidence_metrics['coverage_percentage']:.2f}%")
            
            st.markdown("<br>**Componentes do Score de Confiança:**", unsafe_allow_html=True)
            comp = confidence_metrics['component_scores']
            st.write(f"- Consistência (CV): {comp['cv_score']:.1f}%")
            st.write(f"- Simetria: {comp['symmetry_score']:.1f}%")
            st.write(f"- Densidade Absoluta: {comp['density_score']:.1f}%")
            st.write(f"- Concentração: {comp['concentration_score']:.1f}%")
            st.write(f"- Tratamento de Outliers: {comp['outlier_score']:.1f}%")
            st.write(f"- Cobertura: {comp['coverage_score']:.1f}%")
            st.write(f"- Cobertura de Detecção: {confidence_metrics['coverage_percentage']:.2f}%")
            
            st.markdown("<br>**Componentes do Score de Confiança:**", unsafe_allow_html=True)
            comp = confidence_metrics['component_scores']
            st.write(f"- Consistência (CV): {comp['cv_score']:.1f}%")
            st.write(f"- Simetria: {comp['symmetry_score']:.1f}%")
            st.write(f"- Densidade Absoluta: {comp['density_score']:.1f}%")
            st.write(f"- Concentração: {comp['concentration_score']:.1f}%")
            st.write(f"- Tratamento de Outliers: {comp['outlier_score']:.1f}%")
            st.write(f"- Cobertura: {comp['coverage_score']:.1f}%")
        
        with col2:
            # Gráfico de barras para visualizar o intervalo
            fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
            bars = ax_bar.barh(['Contagem Estimada'], [count], color='#0066cc', alpha=0.7)
            ax_bar.axvline(x=confidence_metrics['lower_bound'], color='red', linestyle='--', alpha=0.5, label='Limite Inferior')
            ax_bar.axvline(x=confidence_metrics['upper_bound'], color='red', linestyle='--', alpha=0.5, label='Limite Superior')
            ax_bar.fill_betweenx([0, 1], confidence_metrics['lower_bound'], confidence_metrics['upper_bound'], 
                                 alpha=0.2, color='yellow', label='Intervalo de Confiança')
            ax_bar.set_xlabel('Número de Pessoas')
            ax_bar.set_title('Intervalo de Confiança da Estimativa')
            ax_bar.legend()
            st.pyplot(fig_bar)
    
    with report_tabs[1]:
        if accuracy_metrics:
            st.markdown("### Comparação com Contagem Real")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                accuracy_color = "#28a745" if accuracy_metrics['accuracy'] >= 90 else "#ffc107" if accuracy_metrics['accuracy'] >= 75 else "#dc3545"
                st.markdown(
                    f"""
                    <div style="padding: 15px; border-radius: 8px; background: {accuracy_color}; color: white; text-align: center;">
                        <h3 style="margin:0; font-size: 32px;">{accuracy_metrics['accuracy']:.1f}%</h3>
                        <p style="margin:5px 0 0 0; font-size: 14px;">Acurácia</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col2:
                st.metric(
                    "Erro Absoluto",
                    accuracy_metrics['error_absolute'],
                    help="Diferença absoluta entre a contagem estimada e real"
                )
            
            with col3:
                st.metric(
                    "Erro Percentual",
                    f"{accuracy_metrics['error_percentage']:.2f}%",
                    help="Erro em porcentagem em relação à contagem real"
                )
            
            st.markdown("---")
            
            # Gráfico de comparação
            fig_comp, ax_comp = plt.subplots(figsize=(10, 6))
            categories = ['Contagem Real', 'Contagem Estimada']
            values = [true_count, count]
            colors = ['#28a745', '#0066cc']
            bars = ax_comp.bar(categories, values, color=colors, alpha=0.7)
            
            # Adicionar valores nas barras
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax_comp.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(value)}',
                           ha='center', va='bottom', fontsize=14, fontweight='bold')
            
            ax_comp.set_ylabel('Número de Pessoas')
            ax_comp.set_title('Comparação: Contagem Real vs Estimada')
            ax_comp.grid(axis='y', alpha=0.3)
            
            # Linha conectando as duas barras
            ax_comp.plot([0, 1], [true_count, count], 'r--', alpha=0.5, linewidth=2, label='Diferença')
            ax_comp.legend()
            
            st.pyplot(fig_comp)
            
            st.markdown("---")
            st.markdown("**Análise Detalhada:**")
            
            if accuracy_metrics['error_percentage'] < 5:
                st.success(f"✅ **Excelente precisão!** O modelo acertou com menos de 5% de erro.")
            elif accuracy_metrics['error_percentage'] < 15:
                st.info(f"ℹ️ **Boa precisão!** O modelo apresentou um erro de {accuracy_metrics['error_percentage']:.2f}%.")
            elif accuracy_metrics['error_percentage'] < 30:
                st.warning(f"⚠️ **Precisão moderada.** O erro foi de {accuracy_metrics['error_percentage']:.2f}%.")
            else:
                st.error(f"❌ **Baixa precisão.** O erro foi de {accuracy_metrics['error_percentage']:.2f}%.")
            
            st.write(f"- Contagem Real: **{true_count}** pessoas")
            st.write(f"- Contagem Estimada: **{count}** pessoas")
            st.write(f"- Diferença: **{accuracy_metrics['error_absolute']}** pessoas")
            
        else:
            st.info("💡 Marque a opção 'Tenho a contagem real' e informe o número real de pessoas para ver a análise de acurácia.")
    
    with report_tabs[2]:
        st.markdown("### Relatório Completo da Análise")
        
        # Preparar dados do relatório
        report_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'image_name': uploaded_file.name,
            'predicted_count': int(count),
            'confidence_metrics': {
                'confidence_score': float(confidence_metrics['confidence_score']),
                'estimated_error_margin': float(confidence_metrics['estimated_error_margin']),
                'interval': f"{confidence_metrics['lower_bound']} - {confidence_metrics['upper_bound']}",
                'variance': float(confidence_metrics['variance']),
                'std_dev': float(confidence_metrics['std_dev']),
                'component_scores': {k: float(v) for k, v in confidence_metrics['component_scores'].items()},
                'coverage_percentage': float(confidence_metrics['coverage_percentage'])
            }
        }
        
        if accuracy_metrics:
            report_data['true_count'] = int(true_count)
            report_data['accuracy_metrics'] = {
                'accuracy': float(accuracy_metrics['accuracy']),
                'error_absolute': int(accuracy_metrics['error_absolute']),
                'error_percentage': float(accuracy_metrics['error_percentage'])
            }
        
        # Exibir relatório em formato texto
        st.markdown("**Informações da Análise:**")
        st.json(report_data)
        
        # Botão para download do relatório JSON
        report_json = json.dumps(report_data, indent=2)
        st.download_button(
            label="📥 Baixar Relatório (JSON)",
            data=report_json,
            file_name=f"relatorio_contagem_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        # Relatório em formato texto formatado
        st.markdown("---")
        st.markdown("**Relatório Formatado:**")
        
        report_text = f"""
# Relatório de Contagem de Multidão
**Data/Hora:** {report_data['timestamp']}
**Imagem:** {report_data['image_name']}

## Resultados
- **Contagem Estimada:** {report_data['predicted_count']} pessoas
- **Score de Confiança:** {report_data['confidence_metrics']['confidence_score']:.1f}%
- **Margem de Erro Estimada:** ±{report_data['confidence_metrics']['estimated_error_margin']}%
- **Intervalo Estimado:** {report_data['confidence_metrics']['interval']} pessoas

"""
        
        if accuracy_metrics:
            report_text += f"""
## Análise de Acurácia
- **Contagem Real:** {report_data['true_count']} pessoas
- **Acurácia:** {report_data['accuracy_metrics']['accuracy']:.1f}%
- **Erro Absoluto:** {report_data['accuracy_metrics']['error_absolute']} pessoas
- **Erro Percentual:** {report_data['accuracy_metrics']['error_percentage']:.2f}%

"""
        
        report_text += f"""
## Métricas Técnicas
- **Variância do Mapa:** {report_data['confidence_metrics']['variance']:.4f}
- **Desvio Padrão:** {report_data['confidence_metrics']['std_dev']:.4f}

---
*Relatório gerado automaticamente pelo Sistema de Contagem de Multidão Poder360*
"""
        
        st.markdown(report_text)
        
        st.download_button(
            label="📄 Baixar Relatório (TXT)",
            data=report_text,
            file_name=f"relatorio_contagem_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

    # -----------------------------
    # HEATMAP
    # -----------------------------
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("🗺️ Mapa de Densidade")

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(density_map, cmap="jet")
    ax.axis("off")
    ax.set_title("Mapa de Densidade - Distribuição de Pessoas na Imagem", fontsize=14, pad=10)
    st.pyplot(fig)
