import flask
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = flask.Flask(__name__, template_folder='templates')

df2 = pd.read_csv('./model/tmdb.csv')

tfidf = TfidfVectorizer(stop_words='english',analyzer='word')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df2['soup'])
print(tfidf_matrix.shape)

#construct cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(cosine_sim.shape)

df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()

# create array with all movie titles
all_titles = [df2['title'][i] for i in range(len(df2['title']))]

def get_recommendations(title):
    # Get the index of the movie that matches the title
    idx = indices[title]
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:21]
    # print similarity scores
    print("\n movieId      score")
    for i in sim_scores:
        print(i)

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # return list of similar movies
    return_df = pd.DataFrame(columns=['Title','Homepage'])
    return_df['Title'] = df2['title'].iloc[movie_indices]
    return_df['Homepage'] = df2['homepage'].iloc[movie_indices]
    return_df['ReleaseDate'] = df2['release_date'].iloc[movie_indices]
    return_df['director']=df2['director'].iloc[movie_indices]
    return_df['imdb']=df2['vote_average']
    return_df['budget']=df2['budget']
    return_df['language']=df2['original_language']
    return_df['duration']=df2['runtime']
    return_df['cast']=df2['cast']
    return_df['voteingcount']=df2['vote_count']
    return return_df

# Set up the main route
@app.route('/', methods=['GET', 'POST'])

def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
            
    if flask.request.method == 'POST':
        m_name = " ".join(flask.request.form['movie_name'].title().split())
#        check = difflib.get_close_matches(m_name,all_titles,cutout=0.50,n=1)
        if m_name not in all_titles:
            return(flask.render_template('notFound.html',name=m_name))
        else:
            result_final = get_recommendations(m_name)
            names = []
            homepage = []
            releaseDate = []
            director = []
            imdb = []
            budget = []
            lang = []
            duration = []
            cast = []
            votingcount = []
            for i in range(len(result_final)):
                names.append(result_final.iloc[i][0])
                releaseDate.append(result_final.iloc[i][2])
                director.append(result_final.iloc[i][3])
                imdb.append(result_final.iloc[i][4])
                budget.append(result_final.iloc[i][5])
                lang.append(result_final.iloc[i][6])
                duration.append(result_final.iloc[i][7])
                cast.append(result_final.iloc[i][8])
                votingcount.append(result_final.iloc[i][9])
                if(len(str(result_final.iloc[i][1]))>3):
                    homepage.append(result_final.iloc[i][1])
                else:
                    homepage.append("#")
                

            return flask.render_template('found.html',movie_names=names,movie_homepage=homepage,search_name=m_name, movie_releaseDate=releaseDate, movie_director=director, movie_imdb=imdb, movie_budget=budget, movie_language=lang, movie_duration=duration, movie_cast=cast, movie_vote=votingcount)

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
    #app.run()
