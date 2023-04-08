#!/usr/bin/env python
# coding: utf-8

# In[16]:


from flask import Flask, request, render_template


# In[15]:


import Recommender


# In[10]:


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recRequest')
def recrequest():
    return render_template('recRequest.html')

@app.route('/recommendation', methods=['POST'])
def recommendation():
    movie = request.form['Movie Name']
    recommendations = Recommender.recommendation_engine(movie)

    return render_template('recommendationResult.html', movie=movie, result = [recommendations])


if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




