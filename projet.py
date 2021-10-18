from numpy.lib.function_base import select
import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import geopandas as gpd
import pyproj
import plotly.graph_objs as go
from wordcloud import WordCloud, STOPWORDS

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("<h1 style='text-align: center; color: white;'>PROJET DATA VISUALISATION</h1>", unsafe_allow_html=True)
st.write('##')

st.write('Voici un aperçu des ventes de 2016 à 2020')
col1, col2, col3, col4, col5 = st.columns(5)
    #fonction count_rows pour connaître 
col1.metric("2016", "341 038")
col2.metric("2017", "379 022", "11.14 %")
col3.metric("2018", "373 267", "-1.58 %")
col4.metric("2019", "393 325", "+6.97 %")
col5.metric("2020", "312 428", "-20.61 %")
st.write('##')

data_list=['2020.csv','2016.csv','2018.csv','2019.csv']

choix_data = st.selectbox('Quelle années voulez vous étudier ?',data_list)
DATA_URL = (choix_data)

col_list=['date_mutation','valeur_fonciere','adresse_numero','adresse_nom_voie','code_postal','nom_commune','nombre_lots','type_local','surface_reelle_bati','surface_terrain','longitude','latitude']

nb_data = st.slider(label="Nombres de données :", min_value=1000, max_value=1500000, value=30000, step=1000)


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_data(col_list):
    data = pd.read_csv(DATA_URL, error_bad_lines=False,delimiter = ",", usecols=col_list)
    return data


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_data_2(col_list):
    data_load_state = st.warning('Chargement des données...')
    data = load_data(col_list)
    data_load_state.success("Done! (using st.cache)")
    return data

data = load_data_2(col_list)
data = data.sample(nb_data)
data = data.dropna(subset=['surface_terrain', 'valeur_fonciere','type_local'])
data['prix_carre'] = data['valeur_fonciere']/data['surface_terrain']
data.drop_duplicates()



def main():
    nav = st.sidebar.radio("Projet Data Visualisation",['Notre Projet','Nos données','Carte de nos données','Analyse rapide','Les endroits les plus chers','test'])


    if nav == "Notre Projet":
        st.markdown("<h1 style='text-align: center; color: withe;'>ALVAREZ HUGO</h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; color: withe;'>DS4</h1>", unsafe_allow_html=True)
        st.write('##')

        st.markdown("<h3 style='text-align: left; color: white;'>Mon compte LinkedIn : </h3>", unsafe_allow_html=True)
        st.markdown('''
            <a href="https://www.linkedin.com/in/hugoalvarez0/">
                <img src="https://i.ibb.co/XsqvvmB/LinkedIn.gif" width="150px" />
            </a>''',
                        unsafe_allow_html=True
                        )
        st.write('##')
        st.markdown("<h3 style='text-align: center; color: [200, 30, 0, 160];'>Étudions les zones géographiques les plus chères de France. </h3>", unsafe_allow_html=True)
        st.image('téléchargement.jpg', width=800)

    if nav == "Nos données":

            select_df = pd.DataFrame()
            select_df = data.mask(data['prix_carre']<150000)
            select_df = select_df.dropna()

            st.markdown("<h2 style='text-align: left; color: white;'>Nos données : (MAX 1 000 000 DATA) </h2>", unsafe_allow_html=True)
            if st.checkbox('Afficher notre DataFrame'):
                st.write(data.tail(nb_data))
            if st.checkbox('Afficher la description de notre DataFrame'):
                st.write(data.describe())
            if st.checkbox('Afficher le dataframe de la valeur foncière la plus élevée'):
                st.write(select_df)       

    if nav == "Carte de nos données":

                st.markdown("<h2 style='text-align: left; color: white;'>Carte de nos données : </h2>", unsafe_allow_html=True) 
                map=pd.DataFrame()
                map['latitude'] = data['latitude']
                map['longitude'] = data['longitude']

                map = map.dropna()

                st.pydeck_chart(pdk.Deck(
                    map_style='dark',
                    initial_view_state=pdk.ViewState(
                    latitude=48.8534,
                    longitude=2.3488,
                    zoom=5,
                    pitch=30,
                    width=700,
                    height=510
                    ),
                    layers=[
                        pdk.Layer(
                            'ScatterplotLayer',
                            data=map,
                            get_position="[longitude, latitude]",
                            get_color='[200, 30, 0, 160]',
                            get_radius=1000,
                            get_size=1
                        ),
                ],       
            ))  
                map['prix_carre']=data['prix_carre']
                maping=map['prix_carre']
                fig = px.density_mapbox(map, lat='latitude', lon='longitude', z=maping, radius=10,
                center=dict(lat=48.8534, lon=2.3488), zoom=5, 
                mapbox_style="dark",width=750,height=600)
                st.plotly_chart(fig)

    if nav == "Analyse rapide":
        st.cache()
        st.write('#') 
        '''  Regardons d'abord les communes disposant du plus de données afin de savoir où mener notre étude :'''
        text = data['nom_commune']
        wordcloud = WordCloud(
            width = 3000,
            height = 2000,
            background_color = 'black',
            stopwords = STOPWORDS).generate(str(text))
        fig = plt.figure(
            figsize = (40, 30),
            facecolor = 'k',
            edgecolor = 'k')
        plt.imshow(wordcloud, interpolation = 'bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.show()
        st.pyplot(fig)

        st.write('##') 
        '''  Nous avons créé une colonne "prix carré" qui stocke le calcul du prix au mètre carré des places '''

        select_df = pd.DataFrame()
        select_df['valeur_fonciere']=data['valeur_fonciere']
        select_df['prix_carre'] = data['prix_carre']
        st.line_chart(select_df)

        select_df = pd.DataFrame()
        select_df['surface_terrain']=data['surface_terrain']
        select_df['surface_reelle_bati'] = data['surface_reelle_bati']
        st.line_chart(select_df)

        st.write('##') 
        '''On peut observer que notre prix au mètre carré évolue en fonction de la valeur du terrain mais aussi que la valeur du terrain dépend principalement de la surface bâtie sur le terrain.'''

        st.write('#')
        ''' '''
        data.plot(kind='scatter', x='longitude', y='valeur_fonciere')
        plt.show()
        st.pyplot()
        st.write('##') 
        ''' Enfin, en se basant sur le dataframe des lieux les plus chers, on remarque qu'ils sont majoritairement concentrés à Paris (correspondant au pique dans le graphique)'''

    if nav == "Les endroits les plus chers":
        option = ['type_local','date_mutation']
        select_hist = pd.DataFrame()
        select_hist['type_local']=data['type_local']
        select_hist['date_mutation']=data['date_mutation']

        hist_x = st.selectbox("Histogram variable", options=option, index=select_hist.columns.get_loc("type_local"))
        
        hist_fig = px.histogram(data, x=hist_x)
        hist_fig = px.histogram(data, x=hist_x, title="Histogram of " + hist_x,template="plotly_white")
        st.write(hist_fig) 
        if hist_x == 'date_mutation' :
            st.markdown("<p style='text-align: left; color: [200, 30, 0, 160];'>On remarque ici que les transactions ont principalement lieu en semaine et non le week-end.</p>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: left; color: white;'>Les endroits les plus chers :</h2>", unsafe_allow_html=True)

        corr_df = data.corr(method='pearson')

        
        select_data = pd.DataFrame()
        select_data = data.mask(data['prix_carre']<150000)
        select_data = select_data.dropna()
        st.write(select_data.tail(10000))
    

        select_df = pd.DataFrame()
        select_df['type_local'] = data['type_local']
        select_df['date_mutation'] = data['date_mutation']
        select_df['prix_carre'] = select_data['prix_carre']
        select_df = select_df.dropna()

        
        st.write('#') 
        st.markdown("<p style='text-align: left; color: [200, 30, 0, 160];'>Regardons le type de local qui est en moyenne le plus cher.</p>", unsafe_allow_html=True)
        
        STOPWORDS.add('ou')
        STOPWORDS.add('dtype')
        STOPWORDS.add('object')
        STOPWORDS.add('Name')
        STOPWORDS.add('type_local')
        STOPWORDS.add('assimilé')
        STOPWORDS.add('commercial')
        STOPWORDS.add('industriel')
        STOPWORDS.add('Length')

        text = select_df['type_local']
        wordcloud = WordCloud(
            width = 3000,
            height = 2000,
            background_color = 'black',
            stopwords = STOPWORDS).generate(str(text))
        fig = plt.figure(
            figsize = (40, 30),
            facecolor = 'k',
            edgecolor = 'k')
        plt.imshow(wordcloud, interpolation = 'bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.show()
        st.pyplot(fig)

        st.write('#') 
        st.markdown("<p style='text-align: left; color: [200, 30, 0, 160];'>Enfin nous avons représenté les endroits où c'est le plus cher.</p>", unsafe_allow_html=True)
        
        select_df=select_df.drop(columns='type_local')
        select_df['prix_carre'].dropna()

        st.pydeck_chart(pdk.Deck(
                    map_style='dark',
                    initial_view_state=pdk.ViewState(
                    latitude=46.98956,
                    longitude=3.159,
                    zoom=5,
                    pitch=50,
                    width=700,
                    height=510
                    ),
                    layers=[
                        pdk.Layer(
                            'ColumnLayer',
                            data=select_data,
                            get_position="[longitude, latitude]",
                            get_color='[200, 30, 0, 160]',
                            get_radius=100000,
                            get_elevation = 15000,
                            get_size=1
                            
                        ),
        
                ],

            )
        )

    if nav == "test":
        map = pd.DataFrame()
        map['location'] = data['latitude'],'/',data['longitude']
        map['prix_carre'] = data['prix_carre']

        fig = px.choropleth_mapbox(map, locations='location', color='prix_carre', color_continuous_scale="Viridis",
                           range_color=(0, 12),
                           mapbox_style="carto-positron",
                           zoom=5, center = {"lat": 48.8534, "lon": 2.3488},
                           opacity=0.5,
                           labels={'prix_carre':'prix au m2'}
                          )
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig)
        
if __name__ == "__main__":
    main()



#correlation matrix
#histogram
#linear chart
#df.plot
#mets quelques pcolonnes seulement et voit si tu peux expliquer ce que t'affiche genre le salaire dépend du niveau de vie etc...

