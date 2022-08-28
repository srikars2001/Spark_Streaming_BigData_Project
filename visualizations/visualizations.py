import matplotlib.pyplot as plt
import pickle
import numpy as np

#getting data using pickle

with open('spam.pkl','rb') as f1:
    spam_count_viz=pickle.load(f1)
    ham_count_viz=pickle.load(f1)

total_spam=sum(spam_count_viz)
total_ham=sum(ham_count_viz)
total_count=total_spam+total_ham
#pie chart 
labels=['Spam','Ham']
sizes=[total_spam/total_count,total_ham/total_count]
explode=[0,0.1]
colors=['#99ff99','#ff9999']
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.tight_layout()
plt.savefig("Spam_and_Ham_pie_chart")
plt.close(fig1)


#line charts multiple 

batches=list(range(len(spam_count_viz)))
""" print(len(spam_count_viz))
print(batches) """
spam_chart1 = plt.plot(batches, spam_count_viz, color='Red')
ham_chart2 = plt.plot(batches, ham_count_viz, color='Blue')
plt.xlabel('Batch',color="green")
plt.ylabel('Count',color="green")
plt.title('Spam/Ham per batch')
plt.legend(['Spam', 'Ham'], loc=3)
plt.savefig("Spam_Ham_per_batch")
plt.close()


# side by side bar chart
plt.rcParams["figure.figsize"] = [13.00, 5]
plt.rcParams["figure.autolayout"] = True
spam_count_bar_chart = spam_count_viz[1:]
ham_count_bar_chart = ham_count_viz[1:]
labels = list(range(len(spam_count_viz)))[1:]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, spam_count_bar_chart, width, label='Spam')
rects2 = ax.bar(x + width / 2, ham_count_bar_chart, width, label='Ham')

ax.set_ylabel('Count')
ax.set_xlabel('Batch')
ax.set_title('Spam/Ham')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
   for rect in rects:
      height = rect.get_height()
      ax.annotate('{}'.format(height),
         xy=(rect.get_x() + rect.get_width() / 2, height),
         xytext=(0, 3), # 3 points vertical offset
         textcoords="offset points",
         ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
plt.savefig("Spam_Ham_per_batch_side_by_side")
#plt.show()
plt.close()



