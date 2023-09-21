# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 13:55:45 2023

@author: DUQUEYROIX
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import OneHotEncoder


def accueil():
    title_style = "color: red; text-align: center;"
    st.markdown(f"<h1 style='{title_style}'>Temps de Réponse de la Brigade des Pompiers de Londres</h1>", unsafe_allow_html=True)
    st.image('./firetruck.jpg')
   

def presentation():
    st.write("### Présentation")
    
    st.write("Le temps de réponse de la Brigade des Pompiers de Londres.")
    st.image("./LFB.png")
    st.write("La brigade des pompiers de Londres, ou London Fire Brigade", "\n", " - 5ème plus grand corps de sapeurs-pompiers au monde", "\n", "- plus de 5 096 sapeurs-pompiers professionnels", "\n", "- 103 casernes", "\n", "- 33 'boroughs',  subdivisions administratives de Londres")
    st.write ("\n")
    st.write ("\n")
    st.write("Notre objectif : analyser leur temps de réponse et prédire le délai d'intervention")


def jeudedonnee():
    title_style = "color: red;"  # Style CSS pour la couleur rouge
    st.write("### Exploration des données")
    
    
    st.markdown(f"<h4 style='{title_style}'>Travaux préparatoires</h4>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)  # Vous pouvez aussi utiliser st.columns()
    col1.write("**Importation des datasets**")
    col1.write("- les incidents dans Londres depuis 2009")
    col1.write("- les mobilisations pour ces incidents")
    col1.write("Disponible sur le [site des pompiers de Londres](https://data.london.gov.uk/dataset/london-fire-brigade-incident-records)")
    col2.write("**Organisation des données**")
    col2.write("- Fusion des deux Dataframes")
    col2.write("-  2.142.544 lignes / 58 colonnes")
    col2.write("- Choix de restriction : années 2017-2022")
    col2.write("-  942.502 lignes / 58 colonnes")
    col3.write("**Nettoyage des données**")
    col3.write("- Suppression des colonnes inutiles au projet")
    col3.write("- Traitement des valeurs manquantes")
    if col3.checkbox("Valeurs manquantes"):
        col3.image("valeurmanquante.png")
        col3.image("valeur2.png")
    col3.write("- Vérification et correction du format des données")    
    
    st.markdown(f"<h4 style='{title_style}'>Données clés</h4>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)  # Vous pouvez aussi utiliser st.columns()
    col1.write("- 941.660  incidents entre 2017 et 2022, datés et minutés")
    col2.write("- 3 types d'incidents : Feux, Fausse alarme et Service Special")
    col3.write("- Interventions sur les 33 arrondissements de Londres")
    
    st.markdown("<h9>Aperçu du Dataframe:</h9>", unsafe_allow_html=True)
    pompier = pd.read_csv("pompier.csv")
    st.dataframe(pompier.head())


def preprocessing(): 
    st.write("### Pre-processing")
    title_style = "color: red;"  # Style CSS pour la couleur rouge
    st.markdown(f"<h4 style='{title_style}'>Enrichissement des données : La Distance </h4>", unsafe_allow_html=True)
    st.write("• Conversion de coordonnées en latitude et longitude pour localiser les incidents")
    st.write("• Ajout de la localisation des casernes (latitude, longitude et adresse)")
    st.write("• Calcul de la distance entre la caserne et le lieu de l'incident: méthode Manhattan")
    
    st.write("**Distribution de la variable distance**")
    st.image("./dist.png")
    
    st.markdown(f"<h4 style='{title_style}'>Traitement des outliers </h4>", unsafe_allow_html=True)
    st.write("• Elimination des distances aberrantes ou extrêmes")
    st.write("• Délimitation de seuils minis et maxis pour les délais d'intervention")
    
    
    st.write("**Délai d'intervention en fonction de la distance**")
    col1, col2 = st.columns(2)
    col1.image("dist_delai1.png", caption="Avant")
    col2.image("dist_delai2.png", caption="Après")
    
def dataviz():
    st.write("### Datavisualisation")
    delai_seconds = 349
    delai_minutes = delai_seconds // 60
    delai_secondes_residuelles = delai_seconds % 60
    st.write("Le délai d'intervention moyen est de :", f"<span style='color: red;'>{delai_seconds} secondes</span> soit", f"<span style='color: red;'>{delai_minutes} min {delai_secondes_residuelles} secondes</span>", unsafe_allow_html=True)
    if st.checkbox("Distribution de la variable DelaiIntervention"):
        st.image("./distribution.png")
    
    pompier_group_year = pd.read_csv("pompier_group_year.csv")
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=pompier_group_year['CalYear'], y=pompier_group_year['DelaiIntervention'], line=dict(color='red')))
    fig5.update_layout(title="Graphique 1 - Délai d'intervention selon l'année", xaxis_title='Années', yaxis_title="Délai d'intervention moyen")
    st.plotly_chart(fig5)
    
    pompier_group_mois= pd.read_csv("pompier_group_mois.csv")
    fig6 = go.Figure()
    fig6.add_traces(go.Scatter(x= pompier_group_mois['Month'], y=pompier_group_mois['DelaiIntervention'], line=dict(color='red')))
    fig6.update_layout(title="Graphique 2 - Delai d'intervention selon le mois", xaxis_title='Mois', yaxis_title="Délai d'intervention moyen")
    st.plotly_chart(fig6)
    
    pompier_group_heures = pd.read_csv("pompier_group_heures.csv")
    fig7 = go.Figure()
    fig7.add_traces(go.Scatter(x= pompier_group_heures['HourOfCall'], y=pompier_group_heures['DelaiIntervention'], line=dict(color='red')))
    fig7.update_layout(title="Graphique 3 - Delai d'intervention selon l'heure", xaxis_title='Heures', yaxis_title="Délai d'intervention moyen")
    st.plotly_chart(fig7)
    
    st.write("Le délai d'intervention dépend du :", f"<span style='color: red;'>type d'incidents </span>. Le délai moyen est le plus long pour les feux, et le plus court pour les autres services ('Special Service') ", unsafe_allow_html=True)

    st.write("**Graphique 4 - Délai d'intervention en fonction de l'arrondissement**")
    st.image('./delai.png')

    
def modelisation():
    st.write("### Modélisation")
    # Mettez ici votre code de modélisation
    st.write("Après le prétraitement de nos données, nous allons nous intéresser à la modélisation.")
    st.write("Deux approches sont étudiées.")
    
    section_choice = st.selectbox("Choix d'une approche",["Modélisation par Regression", "Modélisation par Classification"])
    
    if section_choice == "Modélisation par Regression":
        accueil_regression()  
    elif section_choice == "Modélisation par Classification":
        modelisation_classification()  

def accueil_regression():

    st.write("On cherche à prédire le délai d'intervention.")
    st.markdown("<p style='color: red; font-weight: bold;'>Méthodologie</p>", unsafe_allow_html=True)
    st.write("La variable cible est contenue dans la colonne DelaiIntervention de notre Dataframe")
    st.write("Aperçu du dataframe: ")
    regression= pd.read_csv("df5.csv")
    st.dataframe(regression.head())

    st.markdown("<p style='color: red; font-weight: bold;'>Modélisation</p>", unsafe_allow_html=True)
    st.write("Séparation en deux jeux: 80/20.")
    st.write("4  Modèles de regression testés. ")
    
    section_choice = st.selectbox("Choisir un modèle", ["LinearRegression","Arbre de décision", "RandomForest", "GradientBoostingRegressor"])

    if section_choice == "LinearRegression":
        st.image('linearregression.png')
    elif section_choice == "Arbre de décision":
        st.image('decisiontree.png')
    elif section_choice == "RandomForest":
        st.image("randomforest.png")
    elif section_choice == "GradientBoostingRegressor":
        st.image('gradienbosting.png')
    
    st.write ("Modèle de régression linéaire : **performances modérées**")
    st.write("Modèle de régression par arbre de décision : **overfitting** ")
    st.write("Modèle de régression par forêt aléatoire : **overfitting**")   
    st.write("Modèle de régression par Gradient Boosting: **performances relativement bonnes et cohérentes**")
    st.write ("\n")
    st.write("Comparatif de métriques des deux modèles retenus :")
    st.image('linearregvsgradient.png')
    st.write("- les valeurs sont assez proches entre les ensembles d'entraînement et de test pour les deux modèles","\n", "- les valeurs absolues des erreurs (MAE) et les erreurs quadratiques (MSE, RMSE) sont plus faibles pour le modèle de Gradient Boosting Regressor")
    st.write("Meilleur modèle : **Gradient Boosting Regressor**")
    st.write ("\n")  
    st.write("Variable qui influence le plus le délai d'intervention : **distance**.")
    
    st.markdown("<p style='color: red; font-weight: bold;'>Amélioration du modèle</p>", unsafe_allow_html=True)
    st.write("Amélioration du modèle à l'aide de GridSearchCV et d'une Validation Croisée.")
    st.write("Meilleurs hyperparamètres :","\n","- learning rate : 0.2", "\n","- max depth : 5", "\n", "- n estimators : 150" )
    st.write("Meilleur score (moyenne de la validation croisée) : 0.515")
    st.write("Résultats obtenus : ")
    st.write("- Un coefficient de détermination  de 0.5147198858219273")    
    st.write("- MAE: 56.26258229196011")    
    st.write("- MSE: 5329.878653201203") 
    st.write("- RMSE: 73.00601792456018") 

    st.write(" Un MAE de 56.26 : les prédictions du modèle sont écartées d'environ 56.26 unités(secondes) par rapport aux valeurs réelles.")
    st.write("Un MSE de 5329.87 : les erreurs individuelles sont en moyenne élevées et  le carré de ces erreurs est important.")
    st.write("Un RMSE de 73.006 : les erreurs de prédiction ont tendance à être importantes.")

    st.write("**Conclusion:**  modèle de régression avec des performances modérées.")

def modelisation_classification():

    st.write("Nous allons nous intéresser à la modélisation par classification.")
    st.markdown("<p style='color: red; font-weight: bold;'>Méthodologie</p>", unsafe_allow_html=True)
    
    st.write("La variable cible est contenue dans la colonne 'New_DelaiIntervention' de notre Dataframe")
    st.write(" 1 : délai inférieur ou égal à 250s, 0: délai supérieur à 250s")
    st.write("Aperçu du dataframe :")
    classification= pd.read_csv("df.csv")
    st.dataframe(classification.head())

    st.write("Déséquilibre de classe:")
    st.image("classdistribution.png")


    st.markdown("<p style='color: red; font-weight: bold;'>Modélisation</p>", unsafe_allow_html=True)
    st.write("Séparation en deux jeux: 80/20.")
    st.write("3 Modèles de classification testés.")
    section_choice = st.selectbox("Choisir un modèle", ["Arbre de décision", "RandomForest", "LogisticRegression"])

    if section_choice == "Arbre de décision":
        st.write("Coefficient de détermination du modèle sur le jeux d'entrainement: 0.9983723073204629") 
        st.write("Coefficient de détermination du modèle sur le jeux test: 0.8011737426074609")
    elif section_choice == "RandomForest":
        st.write("Coefficient de détermination du modèle sur le jeux d'entrainement: 0.9983515363001904")
        st.write("Coefficient de détermination du modèle sur le jeux test 0.8435160917543448")
    elif section_choice == "LogisticRegression":
        st.write("Coefficient de détermination du modèle sur le jeux d'entrainement: 0.7977053687422581")
        st.write("Coefficient de détermination du modèle sur le jeux test: 0.7968534030227271")
    st.write ("\n")    
    st.write("**Comparaison des 3 modèles selon les autres métriques**")
    col1, col2, col3 = st.columns(3)  
    col1.write("**Arbre de décision**")
    col1.image("decisiontreeclass.png")
    col2.write("**RandomForest**")
    col2.image("randomforestclass.png")
    col3.write("**LogisticRegression**")
    col3.image("logisticclass.png")
    st.write ("\n")
    st.write("**Meilleur modèle**: Random Forest : ")
    st.write("•accuracy la plus élevée")
    st.write("•AUC-ROC (Area Under the Receiver Operating Characteristic Curve) la plus haute")
    st.write("•AUC-PR (Area Under the Precision-Recall Curve) la plus haute")
    st.write("•scores F1 : meilleurs scores pour les deux classes")
    st.write("•coefficient de détermination relativement élevés")
    

def conclusion():
    st.write("### Conlusion")   
    st.write("Nous avons donc analysé les temps de réponse de la Brigade des Pompiers de Londres")
    st.write ("Prédiction du temps d'intervention selon 2 approches :" ,"\n"," - une régression", "\n", "- une classification")
    st.write ("L'approche par classification est celle qui nous a permis d'avoir de meilleurs prédictions")
    st.write ("Autres pistes à explorer :","\n","- les ressources", "\n", "- les conditions de circulation", "\n", "- le temps passé en intervention")
    st.write("Et encore beaucoup d'autres modèles à évaluer et à améliorer !")
        
pages = ["Accueil","Présentation du projet", "Exploration des données","Pré-processing", "Datavisualisation", "Modélisation", "Conclusion"]

st.sidebar.title("Sommaire")
selected_page = st.sidebar.radio("Aller vers", pages)

# Gestion de la navigation
if selected_page == "Accueil":
    accueil()
elif selected_page == "Présentation du projet":
    presentation()
elif selected_page == "Exploration des données":
    jeudedonnee()
elif selected_page == "Pré-processing":
    preprocessing()  
elif selected_page == "Datavisualisation":
    dataviz()
elif selected_page == "Modélisation":
    modelisation()
elif selected_page == "Prédiction":
    prediction()
elif selected_page == "Conclusion":
    conclusion()
    
    
auteurs = ["Angeline Duqueyroix", "Aurélie Patron", "Soulayman Traboulsi"]

st.sidebar.title("Auteurs:")
for auteur in auteurs:
    st.sidebar.write(auteur)

