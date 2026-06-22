from gensim.downloader import load

print("glove model is loading....")
model=load("glove-wiki-gigaword-50")

result=model.most_similar(positive=['king','women'],negative=['man'], topn=1)
print("king-man+women=",result[0][0],"| similarity:",result[0][1])

result=model.most_similar(positive=['paris','italy'],negative=['france'],topn=1)
print("paris-france+italy=",result[0][0],"|similarity:",result[0][1])

result=model.most_similar(positive=['programming'],topn=5)
print("top 5 words tp programming")
for word,similarity in result:
    print(word,similarity)
