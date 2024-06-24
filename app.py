import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from streamlit_echarts import st_echarts
import numpy as np
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les modèles
clf = joblib.load('modele_fraud_taux.pkl')
clf2 = joblib.load('modele_fraud_cotisation.pkl')

# Configuration de la page
st.set_page_config(page_title="Détection d'Anomalies de Paie", layout="wide")

# Fonction pour générer le rapport PDF
def generate_report(data, num_errors, error_rate, threshold):
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()

        # Titre
        story.append(Paragraph("Rapport de Détection d'Anomalies de Paie", styles['Title']))
        story.append(Spacer(1, 12))

        # Résumé
        story.append(Paragraph(f"Nombre total de bulletins analysés : {len(data)}", styles['Normal']))
        story.append(Paragraph(f"Nombre de bulletins avec anomalies : {num_errors}", styles['Normal']))
        story.append(Paragraph(f"Taux d'anomalies : {error_rate:.2f}%", styles['Normal']))
        story.append(Paragraph(f"Seuil de confiance pour les anomalies : {threshold}", styles['Normal']))
        story.append(Spacer(1, 12))

        # Distribution des Anomalies
        story.append(Paragraph("Distribution des Anomalies", styles['Heading2']))
        fig, ax = plt.subplots(figsize=(6, 4))
        data['anomaly'].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%')
        plt.title('Répartition des Anomalies')
        plt.axis('equal')
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        story.append(Image(img_buf, width=6*inch, height=4*inch))
        plt.close(fig)
        story.append(Spacer(1, 12))

        # Distribution des Probabilités de Fraude
        story.append(Paragraph("Distribution des Probabilités de Fraude", styles['Heading2']))
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=data, x='prediction_fraud', hue='anomaly', kde=True, ax=ax)
        plt.title('Distribution des Probabilités de Fraude')
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        story.append(Image(img_buf, width=6*inch, height=4*inch))
        plt.close(fig)
        story.append(Spacer(1, 12))

        # Matrice de Corrélation
        story.append(Paragraph("Matrice de Corrélation (Colonnes Numériques)", styles['Heading2']))
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr = data[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        plt.title('Matrice de Corrélation')
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight')
        img_buf.seek(0)
        story.append(Image(img_buf, width=7*inch, height=5*inch))
        plt.close(fig)
        story.append(Spacer(1, 12))

        # Top 10 des anomalies les plus probables
        top_anomalies = data[data['anomaly'] == 'Anomaly'].sort_values('prediction_fraud', ascending=False).head(10)
        story.append(Paragraph("Top 10 des anomalies les plus probables :", styles['Heading2']))
        for index, row in top_anomalies.iterrows():
            story.append(Paragraph(f"ID: {index}, Probabilité de fraude: {row['prediction_fraud']:.2f}", styles['Normal']))

        # Générer le PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Erreur lors de la génération du rapport PDF: {e}")
        return None

# Sidebar pour les options
st.sidebar.title("Options")
show_raw_data = st.sidebar.checkbox("Afficher les données brutes")
threshold = st.sidebar.slider("Seuil de confiance pour les anomalies", 0.0, 1.0, 0.5)

# Titre principal avec style
st.markdown("""
    <h1 style='text-align: center; color: #1E90FF;'>Détection d'Anomalies dans les Données de Paie</h1>
    """, unsafe_allow_html=True)

# Télécharger le fichier CSV
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)

    # Préparer les données et faire des prédictions
    X_new = new_data.drop('fraud_taux', axis=1, errors='ignore')
    new_data['prediction_taux'] = clf.predict(X_new)

    X_new_2 = new_data.drop('fraud', axis=1, errors='ignore')
    new_data['prediction_fraud'] = clf2.predict_proba(X_new_2)[:, 1]  # Probabilités

    # Appliquer le seuil
    new_data['anomaly'] = new_data['prediction_fraud'].apply(lambda x: 'Anomaly' if x > threshold else 'Normal')

    # Afficher les statistiques résumées
    col1, col2 = st.columns(2)
    with col1:
        num_errors = (new_data['anomaly'] == 'Anomaly').sum()
        st.metric(label="Bulletins de paie avec anomalies", value=num_errors)
    with col2:
        error_rate = (num_errors / len(new_data)) * 100
        st.metric(label="Taux d'anomalies", value=f"{error_rate:.2f}%")

    # Afficher les données avec les prédictions
    if show_raw_data:
        st.write('**Données avec les prédictions et les anomalies :**')
        st.dataframe(new_data.style.applymap(lambda x: 'background-color: rgba(255, 0, 0, 0.3)' if x == 'Anomaly' else '', subset=['anomaly']))

    # Graphique de distribution des anomalies
    st.write('**Distribution des Anomalies :**')
    fig_pie = px.pie(new_data, names='anomaly', title='Répartition des Anomalies')
    st.plotly_chart(fig_pie)

    # Graphique de densité pour la probabilité de fraude
    st.write('**Distribution des Probabilités de Fraude :**')
    fig_histogram = px.histogram(new_data, x='prediction_fraud', color='anomaly', 
                                 marginal="box", hover_data=new_data.columns)
    st.plotly_chart(fig_histogram)

    # Heatmap des corrélations
    st.write('**Matrice de Corrélation (Colonnes Numériques) :**')
    numeric_cols = new_data.select_dtypes(include=[np.number]).columns
    corr = new_data[numeric_cols].corr()
    fig_heatmap = px.imshow(corr, text_auto=True, aspect="auto")
    st.plotly_chart(fig_heatmap)

    # Téléchargement des résultats
    csv = new_data.to_csv(index=False)
    st.download_button(
        label="Télécharger les résultats CSV",
        data=csv,
        file_name="resultats_anomalies.csv",
        mime="text/csv",
    )

    # Graphique interactif avec ECharts
    st.write('**Graphique Interactif des Anomalies :**')
    options = {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"data": ["Normal", "Anomaly"]},
        "xAxis": {"type": "value"},
        "yAxis": {"type": "category", "data": numeric_cols.tolist()},
        "series": [
            {
                "name": "Normal",
                "type": "bar",
                "stack": "total",
                "label": {"show": True},
                "emphasis": {"focus": "series"},
                "data": [new_data[new_data['anomaly'] == 'Normal'][col].mean() if pd.api.types.is_numeric_dtype(new_data[col]) else 0 for col in numeric_cols],
            },
            {
                "name": "Anomaly",
                "type": "bar",
                "stack": "total",
                "label": {"show": True},
                "emphasis": {"focus": "series"},
                "data": [new_data[new_data['anomaly'] == 'Anomaly'][col].mean() if pd.api.types.is_numeric_dtype(new_data[col]) else 0 for col in numeric_cols],
            },
        ],
    }
    st_echarts(options=options, height="500px")

    # Bouton pour générer le rapport
    if st.button("Générer le rapport PDF"):
        pdf = generate_report(new_data, num_errors, error_rate, threshold)
        if pdf:
            st.download_button(
                label="Télécharger le rapport PDF",
                data=pdf,
                file_name="rapport_anomalies.pdf",
                mime="application/pdf"
            )
        else:
            st.error("Le rapport PDF n'a pas pu être généré. Veuillez vérifier les données et réessayer.")

else:
    st.info('Veuillez télécharger un fichier CSV pour commencer l\'analyse.')

# Ajout de la section chatbot
st.sidebar.markdown("---")
st.sidebar.header("Chat avec l'Assistant IA")

# Initialiser l'historique du chat s'il n'existe pas
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

user_input = st.sidebar.text_input("Posez votre question ici :")

if user_input:
    # Ajouter la question de l'utilisateur à l'historique
    st.session_state.chat_history.append(f"Vous : {user_input}")

    # Logique de réponse améliorée
    user_input_lower = user_input.lower()
    if "anomalie" in user_input_lower or "fraude" in user_input_lower:
        response = "Les anomalies ou fraudes potentielles sont détectées en comparant les données avec des modèles préétablis. Nous utilisons des techniques d'apprentissage automatique pour identifier les schémas inhabituels dans les données de paie."
    elif "seuil" in user_input_lower:
        response = f"Le seuil actuel pour les anomalies est fixé à {threshold}. Ce seuil peut être ajusté dans les options pour modifier la sensibilité de la détection."
    elif "données" in user_input_lower or "fichier" in user_input_lower:
        response = "Pour analyser vos données, vous devez télécharger un fichier CSV contenant les informations de paie. Assurez-vous que le fichier est correctement formaté avant de le télécharger."
    elif "résultat" in user_input_lower or "analyse" in user_input_lower:
        response = "Les résultats de l'analyse sont présentés sous forme de graphiques et de tableaux. Vous pouvez voir la distribution des anomalies, les probabilités de fraude, et télécharger un rapport détaillé."
    elif "aide" in user_input_lower or "utiliser" in user_input_lower:
        response = "Pour utiliser cette plateforme, commencez par télécharger votre fichier CSV. Ensuite, ajustez le seuil de détection si nécessaire. Vous pouvez explorer les différents graphiques et télécharger les résultats. N'hésitez pas à me poser des questions spécifiques sur chaque fonction."
    elif "rapport" in user_input_lower or "pdf" in user_input_lower:
        response = "Vous pouvez générer un rapport PDF détaillé en cliquant sur le bouton 'Générer le rapport PDF' après avoir analysé vos données. Ce rapport contiendra un résumé des anomalies détectées et d'autres informations pertinentes."
    else:
        response = "Je ne suis pas sûr de comprendre votre question. Pouvez-vous la reformuler ou demander des informations sur les anomalies, le seuil de détection, les données à utiliser, ou comment interpréter les résultats ?"

    # Ajouter la réponse à l'historique
    st.session_state.chat_history.append(f"Assistant : {response}")

# Afficher l'historique du chat
st.sidebar.markdown("---")
st.sidebar.subheader("Historique du chat")
for message in st.session_state.chat_history:
    st.sidebar.text(message)
