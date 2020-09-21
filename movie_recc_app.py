import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from difflib import get_close_matches

# Build a recommender based on the following metadata: the 3 top actors, the director, related genres and the movie plot keywords.


credits = pd.read_csv('./dataset/tmdb_5000_credits.csv')
movies= pd.read_csv('./dataset/tmdb_5000_movies.csv')
movies.rename(columns={'id':'movie_id'}, inplace=True)
df = movies.merge(credits, on='movie_id', how='left')
df.drop(['tagline','title_x','title_y'],axis=1, inplace=True) # Drop unnecessary features

features = ['cast', 'crew', 'keywords', 'genres']
# These features are in the form of 'strigfied' lists, which requer the literal_eval method to
# get the embedded dictionary

for feature in features:
    df[feature] =df[feature].apply(literal_eval)

# Create the functions to extract the correspondent information from the features
def get_director(feature): # Get the name of th director from the crew feature. If there is none, return NaN
    for i in feature:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def get_list(feature): # Get a list of the top 3 elements or less
    if isinstance(feature,list):
        names = [i['name']for i in feature]
        if len(names) > 3: #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list
            names = names[:3]
        return names
    return []  #Return an empty list in case of missing/malformed data

# Create a new column with the directors' names
df['director'] = df['crew'].apply(get_director)

# Get the suitable form of the following features using the get_list function
features = ['cast', 'keywords','genres']
for feature in features:
    df[feature] = df[feature].apply(get_list)

# TNow, we convert the names and keyword instances into lowercase and strip all the spaces between them. 
# This is done so that the  vectorizer doesn't count the, for example,  Johnny of "Johnny Depp" and "Johnny Galecki" as the same.
def clean_data(feature):
    if isinstance(feature,list):
        return [str.lower(i.replace(' ','')) for i in feature]
    else:
        if isinstance(feature,str):
            return str.lower(feature.replace(' ',''))
        else:
            return ''
# Then, we apply the clean_data function to the features
features = ['cast', 'keywords', 'director','genres']

for feature in features:
    df[feature] = df[feature].apply(clean_data)

# To feed out vectorizes, we need to create a "metadata soup", that is a string that unites
# all the metadata selected to the recommender (actors, director,keywords)

def create_soup(x):
    return ' '.join(x['keywords']) + ' '+ ' '.join(x['cast'])+ ' '+x['director']+ ' '+ ' '.join(x['genres'])

df['soup'] = df.apply(create_soup, axis=1)

# From this, we use CountVectorizer to create a sparse count matrix of text documents,
# in our case, the soup column
count = CountVectorizer(stop_words='english')
count_matrix=count.fit_transform(df['soup'])

# We will be using the cosine similarity to calculate a numeric quantity that denotes the similarity
# between two movies.
cosine = cosine_similarity(count_matrix, count_matrix)

# Now, we create a mapping of the movies and their indices for our recommendation function
indices = pd.Series(df.index,index=df.original_title)

def get_recommendations(title,consine):
    idx = indices[title] # get the movie's index
    sim_scores = list(enumerate(cosine[idx])) # get the pairwise similiarity score of all movies with that movie
    sim_scores = sorted(sim_scores,key=lambda x:x[1], reverse=True) # Sort the movies by the similarity score
    sim_scores = sim_scores[1:11] # get the 10 most similar movies by similarity scores
    movies_indices = [i[0] for i in sim_scores]
    for i in df['original_title'].iloc[movies_indices].values:
        if i is not None:
            st.write(i) # Return the top 10 most similar movies



def main():
    bar = st.sidebar.selectbox(label='Menu', options=['Recommender', 'About the app'])
    if bar is 'Recommender':
        st.title('Movie Recommender System')
        st.image('popcorn.jpg', use_column_width=True)
        m = st.text_input('Choose a movie: \n').title()
        if m:
            if m not in indices.index:
                st.subheader('Did you mean one of these?')
                for i in get_close_matches(m, indices.index, n=3,cutoff=0.5):
                    st.write(i)
            else:
                idx = indices[m]  # get the movie's index
                st.subheader('Top 10 recommended movies similar to {}\n'.format(m))
                get_recommendations(idx, cosine)

    if bar is 'About the app':
        st.subheader('About the app')
        st.markdown(
            """
            Based on the [TMDB 5000 movies dataset](https://www.kaggle.com/tmdb/tmdb-movie-metadata),
            this movie recommender was built using movies metadata, namely the top 3 actors, the director, the related genres and movies plot keywords,
            to generate a list of 10 recommended movies. The recommendation itself derives from the cosine similarity score that the inputed movie
            has in relation to the other movies in the dataset.
            
            Its author is JP Vazquez, an aspiring data scientist. The app's creation was motivated by pedagogical reasons,
            since the knowdledge about a recommendation system is very important to any data scientist's work.
            
            For more, please check out:
            * [LinkedIn](www.linkedin.com/in/joão-pedro-vazquez)
            * [Github](https://github.com/jpvazquezz)""")

if __name__ == '__main__':
	main()

