#%%
import pandas
import numpy
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from matplotlib import cm
from datetime import datetime
import scipy.cluster.hierarchy as shc
import re
#%% Funcations
def CleaningData():
    #making collums for movies and ratings
    collum_ratings = ratings.drop(['timestamp'], axis=1)
    collum_movies = movies.drop(['title'], axis=1)
    newRatingColum = collum_ratings.reindex(columns = ["userId", "rating", "movieId"])
    print(collum_ratings.columns)
    print(collum_movies.columns)

    # Changing arrays to list
    movieId_list = collum_movies['movieId'].tolist()
    rattingGenres_list = collum_movies['genres'].tolist()

    #replacing movieId by the relatted genres
    newRatingColum['movieId'] = newRatingColum['movieId'].replace(movieId_list, rattingGenres_list)

    #create list for generes
    uncallsifiedGenre = newRatingColum.loc[newRatingColum.movieId == "(no genres listed)", ["movieId"]]
    print(uncallsifiedGenre)
    print(uncallsifiedGenre.count())
    #Removing the rows that do not have a generes
    cleanRatings = newRatingColum[~newRatingColum.movieId.str.contains("(no genres listed)", ["movieId"])]
    print(cleanRatings)
    # doble checks if any rattings were missed
    remaing = cleanRatings.loc[cleanRatings.movieId == "(no genres listed)", ["movieId"]]
    len(remaing)
    # export ratings to csv so we dont have to redo data handling
    cleanRatings.to_csv("Data/Q1/CleanedRatings.csv", index=False)

generes = []

def checkGenres():
    for genre in DataCleaned.movieId:
        x = genre.split("|")
        for y in x:
            if y in generes:
                print(True)
            else:
                generes.append(y)
                print(False)

generes = ["Action", "Adventure", "Animation", "Children", "Comedy",
"Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
"Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western", "IMAX"]
# runs through the cleaned data and looks for genres and adds it to a array
#Making List to save time on the outpur of the search above

def CreateFinalDatabase():
    data_columns = ["userId"]
    for genre in generes:
        data_columns.append(genre)
    
    #Creating variables for proccessing
    users = list(range(1,611)) # create a list of numbers from 1 to 610
    row_ratings = len(DataCleaned)
    # making list formats for ratings & movieId
    user_ratings = DataCleaned['rating'].tolist()
    ratings_genres = DataCleaned['movieId'].tolist()
    final_data = []
    current_user = 0
    current_rating = 0
    current_genre = ""
    
    # Separate each movie into one genre, first one
    for i in range(len(DataCleaned)):
        temp_list = []
        current_user = DataCleaned.userId[i]
        current_rating = DataCleaned.rating[i]
        temp_list.append(current_user)
        temp_list.append(current_rating)
        rated_genres = DataCleaned.movieId[i]
        separated_genre_list = rated_genres.split('|')
        temp_list.append(separated_genre_list[0])
        final_data.append(temp_list)
    inital_user_rating = []
    for user_i in range(len(users)):
        temp_list1 = []
        temp_list1.append(user_i+1) #current user
        for col in generes:
            temp_list1.append(0.0) # makes the user ratings 0.0 for all users
        inital_user_rating.append(temp_list1)

    genre_map = [("Action", "1"), ("Adventure", "2"), ("Animation", "3"), ("Children", "4"), 
        ("Comedy", "5"), ("Crime", "6"), ("Documentary", "7"), ("Drama", "8"), ("Fantasy", "9"), 
        ("Film-Noir", "10"), ("Horror", "11"), ("Musical", "12"), ("Mystery", "13"), ("Romance", "14"),
        ("Sci-Fi", "15"), ("Thriller", "16"), ("War", "17"), ("Western", "18"), ("IMAX", "19")]

    converted_rating = []
    for i in range(len(final_data)):
        current_user = final_data[i][0]
        current_rating = final_data[i][1]
        current_genre = final_data[i][2]
        for k, v in genre_map:
            current_genre = current_genre.replace(k, v)
        converted_rating.append([current_user, current_rating, int(current_genre)])

    rating_listFormat = DataCleaned.values.tolist()  
    copy_rating = converted_rating.copy()
    finished_indices = []
    index_n = 0
    average_rating_num = 0.0

    for user in range(len(users)):
        current_user = user+1 # get current user
        # Current user rating for each genre
        current_user_ratings = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #  Current rating count for each genre
        current_rating_count = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        user_n_rating = []
        user_n_rating.append(current_user)

        for col in generes:
            user_n_rating.append(0.0)

        for index in range(len(copy_rating)):
            if copy_rating[index][0] > current_user:
                break
            else:
                current_rating = copy_rating[index][1]
                current_genre_num = copy_rating[index][2] - 1
                current_user_ratings[current_genre_num] += current_rating
                current_rating_count[current_genre_num] += 1
                index_n += 1

        for index1 in range(index_n):
            copy_rating.pop(0)
        
        for index2 in range(18):
            if current_rating_count[index2] == 0.0:
                current_rating_count[index2] = 1.0
            average_rating_num += current_user_ratings[index2]/current_rating_count[index2]
        average_rating_num = average_rating_num/18.0

        for index3 in range(18):
            if current_user_ratings[index3] == 0.0:
                current_user_ratings[index3] = average_rating_num

        for index4 in range(18):
            rating_holder = (current_user_ratings[index4]/current_rating_count[index4])
            user_n_rating[index4+1] = round(rating_holder, 2)

        finished_indices.append(user_n_rating)
        index_n = 0
        average_rating_num = 0.0

    final_data_set = pandas.DataFrame(finished_indices, columns = data_columns)
    final_data_set = final_data_set.drop(['IMAX'], axis=1)
    final_data_set.to_csv("final_data_set.csv", index=False)

def combined_datset(ratings, movies):
    ratings = ratings.iloc[:,:3]
    final_combined = []
    temp_movieId = 0
    temp_movieId_index = 0
    ratings_list = ratings.values.tolist()
    movies_list = movies.values.tolist()
    movieId_list = movies.movieId.tolist()
    
    for i in range(len(ratings_list)):
        temp = []
        temp.append(int(ratings_list[i][0]))
        temp.append(ratings_list[i][2])
        temp.append(int(ratings_list[i][1]))

        temp_movieId = int(ratings_list[i][1])
        temp_movieId_index = movieId_list.index(temp_movieId)
        temp.append(movies_list[temp_movieId_index][1])
        temp.append(movies_list[temp_movieId_index][2])
        final_combined.append(temp)
    combinded = pandas.DataFrame(final_combined, columns=["userId", "rating", "movieId", "title", "genres"])
    combinded.to_csv("Combined_Data.csv")

#%% Reading data
movies = pandas.read_csv('./Data/Q1/movies.csv')
ratings = pandas.read_csv('./Data/Q1/ratings.csv')
print(movies.shape)
print(ratings.shape)

#CleaningData() #uncomment to clean the data
DataCleaned = pandas.read_csv('Data/Q1/CleanedRatings.csv')
#checkGenres() #uncomment to create genres
#reateFinalDatabase() #uncomment to create final database
#combined_datset(ratings, movies)


#%%
finalDatabase = pandas.read_csv('./final_data_set.csv')
train = finalDatabase.iloc[:, 1:]
combined_data = pandas.read_csv("./Combined_Data.csv")

# %%
n_clust = 30
kmeans = KMeans(n_clusters=n_clust, random_state=4).fit(train)
kmeans_cluster_Description = kmeans.cluster_centers_
kmeans_cluster_Description = kmeans_cluster_Description.tolist()
K_labels = kmeans.labels_
K_labels = K_labels.tolist()
cluster_indexes = []
for cluster in range(n_clust):
    temp_result = []
    temp_result = [index for index, value in enumerate(K_labels) if value == cluster]
    cluster_indexes.append(temp_result)
gmm = GaussianMixture(n_clust, random_state=5).fit(train)
gmm_labels = gmm.predict_proba(train)
gmm_labels = gmm_labels.tolist()
gmm_cluster_desc = gmm.means_
gmm_cluster_desc = gmm_cluster_desc.tolist()
hac = AgglomerativeClustering(distance_threshold=10, n_clusters=None, affinity='euclidean', linkage='ward').fit(train)
hac_labels = hac.labels_.tolist()
hac_Nclusters = hac.n_clusters_

hac_clusters = []
for cluster in range(18):
	temp_result = []
	temp_result = [i for i, x in enumerate(hac_labels) if x == cluster]
	hac_clusters.append(temp_result)

#%%
cluster_ratings = [] 
def recommended_movies_byCluster(cluster_n):
    temp_rating_list_top40 = []
    curr_index = cluster_indexes[cluster_n]
    temp_result = combined_data[combined_data["userId"].isin(curr_index)]
    order_byViews = temp_result["movieId"].value_counts().index.tolist()
    top40 = order_byViews[:40]
    top40movie_rating = temp_result[temp_result["movieId"].isin(top40)]
    top40movie_rating = top40movie_rating.iloc[:,1:]
    
    temp_rating_list_top40 = top40movie_rating.groupby("movieId")["rating"].mean()
    templist = temp_rating_list_top40.tolist()
    temp_keys = temp_rating_list_top40.keys()
    
    rating_Id = []
    for j in range(len(temp_keys)):
        rating_Id.append((round(templist[j], 2)))
    
    temp_list_top40 = []
    for i in top40:
        temp_list_empty = []
        temp_list_empty.append(0.0)
        temp_list_empty.append(i)
        current_movie_name = top40movie_rating.loc[top40movie_rating["movieId"] == i, "title"].iloc[0]
        current_movie_genre = top40movie_rating.loc[top40movie_rating["movieId"] == i, "genres"].iloc[0]
        temp_list_empty.append(current_movie_name)
        temp_list_empty.append(current_movie_genre)
        temp_list_top40.append(temp_list_empty)
    
    data_temp_list_top40 = pandas.DataFrame(temp_list_top40, columns=["rating", "movieId", "title", "genres"])
    data_temp_list_top40 = data_temp_list_top40.sort_values(by=["movieId"])
    data_temp_list_top40 = data_temp_list_top40.assign(rating=rating_Id)
    data_temp_list_top40 = data_temp_list_top40.sort_values(by=["rating"], ascending=False)
    
    return top40movie_rating, data_temp_list_top40

cluster_ratings, data_top40_movie = recommended_movies_byCluster(3)

# %%
def recommend_user_by_Cluster(recommend_movies, user_n):
    copy_combined = combined_data.copy()
    copy_combined = copy_combined.loc[copy_combined["userId"] == user_n]
    copy_combined = copy_combined.iloc[:,2].tolist()
    print(len(copy_combined))
    new_recommendation = recommend_movies
    print(len(new_recommendation))
    new_recommendation = new_recommendation[~new_recommendation.movieId.isin(copy_combined)]
    return new_recommendation

user_n_recomendation = recommend_user_by_Cluster(data_top40_movie, 1)

generes_x = ["Action", "Adventure", "Animation", "Children", "Comedy",
"Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
"Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
#%%
def eval_user(user_n):
    user = finalDatabase.iloc[user_n-1, 1:]
    kmeans_cluster_n = K_labels[user_n-1]
    gmm_cluster_n = gmm_labels[user_n-1]
    gmm_cluster_n_largest = max(gmm_cluster_n)
    gmm_cluster_n = gmm_cluster_n.index(gmm_cluster_n_largest)
    
    fig = plt.figure(figsize=[20,20])
    ax = fig.add_subplot(3,1,1)
    ax.bar(generes_x, user)
    ax.set_title('User {0} ratings'.format(user_n))

    ax = fig.add_subplot(3,1,2)
    ax.bar(generes_x, kmeans.cluster_centers_[kmeans_cluster_n,:])
    ax.set_title('Kmeans Cluster {0}'.format(kmeans_cluster_n))

    ax = fig.add_subplot(3, 1, 3)
    ax.bar(generes_x, gmm.means_[gmm_cluster_n,:])
    ax.set_title('GMM Cluster {0}'.format(gmm_cluster_n))
    fig.show()

#%%
eval_user(4)
#%%
eval_user(42)
#%%
eval_user(314)
#%%
