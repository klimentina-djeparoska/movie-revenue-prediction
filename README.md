# movie-revenue-prediction

Dataset: https://www.kaggle.com/c/tmdb-box-office-prediction/data

 
TMDB Box Office Prediction

### 1.	Abstraction

Today, movies are getting more and more attention and grow in popularity. From comedy to science fiction they serve people as entertainment in their everyday life.

TMDB Box Office Prediction is a project for trying to predict the revenue of a movie, based on the gathered data available from the TMDB dataset.
Because the problem I will be trying to solve is prediction of a real value for the revenue, it will be solved using regression as part of the supervised learning.


 
### 2.	Choosing features and preprocessing the data

The available features from the Movies database are: 
- 'id', 'belongs_to_collection', 'budget', 'genres', 'homepage', 'imdb_id', 'original_language', 'original_title', 'overview', 'popularity', 'poster_path', 'production_companies', 'production_countries', 'release_date', 'runtime', 'spoken_languages', 'status', 'tagline', 'title', 'Keywords', 'cast', 'crew', 'revenue'. 

Here are the features that were chosen to be used in building the model and a short explanation why and the way it is processed in the code:

-imdb_id: as a unique identifier for the movie given from IMDB, is used for removing duplicates in the dataset

-belongs_to_collection: tells if the movie is part of a collection and which one, this feature is transformed from object to 1 or 0, 1 if it is part of a collection and 0 otherwise, this is an important feature because movies that are part of a collection tend to make more money (ex. famous collections: Harry Potter, Star Wars etc.)

-budget: is the amount of money available for making the movie, it is chosen as a feature because the budget often effects the quality of movie

-genres: is a list of objects stating to which genre the movie belongs (it can be to one or more). This column is removed and at end it is added one column per genre so we can represent the movie that belongs to more genres. The movie has a 0 or 1 for each column of the genres depending whether it belongs or not

-homepage: is the official page of the movie, it is translated to 0 and 1 whether it has or not. The movies that have homepage tend to make more money

-original_language: is the main language used in the movie, most of the time is English as a general most used language. Each language is transformed to number and then that number is convert to binary representation of the number

-popularity: represents the rating by imdb based on the searches and visits.

-production_companies: here there is one or more companies given as objects. If there is more than one production company, then they are ordered based on they id the represents how important and popular that company is. The id number is then translated to binary

-production_countries: the country where the movie is produced in, it is done a mapping for the shortcuts of the countries with numbers and then those numbers are translated to binary

-release_date: the date the movie is released. From here only the year is taken into account. It is divided in periods let say from 1910 to 2020 and so on. Because now we are watching more movies that in the past 

-runtime: the length of time of the movie. It is also divided in intervals a good movie is usually between 90min and 120min

-status: whether the film is released or is still in making

-crew: members of the crew -director, actors etc. Here the number of the crew is taken into account the more people the crew has the better the film will be, so as a result it will make more revenue

-revenue: how much money the movie makes, this is also the column that will be predicted in this problem

The following columns are dropped:
-‘original_title’, 'tagline', 'title', 'Keywords', 'cast', 'overview', 'poster_path', 'spoken_languages'

 

### 3.	The model:

-before I started with building the model, I searched for other problems solved using regression as part of the supervised learning, so from there I decided to use the following techniques 

- Sequential model from Keras library, is used for creating deep learning model

- it has two dense layers with 8 units each

- and one output layer that returns a single continues value

- the dataset is divided to train and test, 0.7 and 0.3 accordingly

a)	for training the model, model.fit function is used also from Keras library, which trains the model for a fixed number of epochs (the iterations on the dataset)

b)	to prevent overfitting the EarlyStopping function is used, her purpose is if there aren’t any improvement in the predictions (in a X number of epochs) to stop the training

c)	for the loss and error rate MSE (Mean squared error) and MAE (Mean absolute error) are used, because they are known to be the most used in regression problems

d)	after evaluation of the model the MAE for the test data is between 1.900 to 5.000 which I think is a good approximation considering there are values of revenue that are around 1.000.000.000



### 4.	Adjusting parameters and testing


-	For the number of neurons used in each processing layer in the model, I used 8 neurons per layer. The reason why I didn’t used more is because I already had 32 features (each genre is represented as a separate column) so adding more neurons would have resulted with the curse of dimensionality


-	By adjusting the number of epochs, I first tried with 1000 and then increased it to 2000, 2000 gave a better prediction with smaller error rate. So, then I tried with 3000, but after some iteration I noticed that the MAE value didn’t decrease any more, no further improvements where made so I decide to set the number of epochs to 2000.


-	I also tried different learning rates for the optimizer. I first tried with 0.001 but I noticed in the logs of the iterations that the error goes back and forth, so I decreased the value several times and end up with 0.00001 learning rate. Because it gave the best results, I try to go even lower that this, but it didn’t give any improvement, so I kept the previous value.


-	All the conclusions were made based on several executed tests and different form of evaluation either from plots from the result and then comparing that with the previous plot or from the logs on each iteration once the algorithm is training the model





### 5.	Summary

The final model build, with all the adjustments and testing done gave the following results:

From the final linear plot for error in the prediction of revenue, we can conclude that the model is good in approximating the revenue the movie will make based on the selected features.

So, as conclusion I would like to say that the revenue one movie would make, can be approximated – predicted if we have enough data in regards to which genre it belongs, budget, crew etc.


Developed by:
Klimentina Djeparoska
