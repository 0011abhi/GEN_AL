import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.downloader import load

print("loading glove model...")
model=load("glove-wiki-gigaword-50")

words=['football','volleyball','soccer','tennis','circket']
vectors=[model[word]for word in words]

pca=PCA(n_components=2)
points=pca.fit_transform(vectors)

plt.figure(figsize=(15,10))
for word,(x,y) in zip (words,points):
    plt.scatter(x,y,color='blue')
    plt.text(x+0.02,y+0.02,words,fontsize=12)
plt.show()

result=model.most_similar("programmming",topn=5)    
for word,score in result:
    print(word,score)
