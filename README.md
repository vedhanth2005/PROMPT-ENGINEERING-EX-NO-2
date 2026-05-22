# EX-02-Cross-Platform-Prompting-Evaluating-Diverse-Techniques-in-AI-Powered-Text-Summarization

## AIM
To evaluate and compare the effectiveness of prompting techniques (zero-shot, few-shot, chain-of-thought, role-based) across different AI platforms (e.g., ChatGPT, Gemini, Claude, Copilot) in a specific task: text summarization.

## SCENARIO:
You are part of a content curation team for an educational platform that delivers quick summaries of research papers to undergraduate students. Your task is to summarize a 500-word technical article on "The Basics of Blockchain Technology" using multiple AI platforms and prompting strategies.

Your goal is to determine which combination of prompting technique + platform provides the best summary in terms of:

Accuracy

Coherence

Simplicity

Speed

User experience

## OUTPUT
import pandas as pd
df=pd.read_csv(r"C:\Users\Downloads\titanic_dataset.csv")
df
<img width="1310" height="504" alt="image" src="https://github.com/user-attachments/assets/8c847aa3-7e0a-4e0e-a666-b587636349ed" />
df.shape
<img width="1209" height="33" alt="image" src="https://github.com/user-attachments/assets/f7f248cc-e682-42e2-a361-21082048646b" />
df.set_index("PassengerId",inplace=True)
df
<img width="1351" height="542" alt="image" src="https://github.com/user-attachments/assets/d4b703d9-e8fc-4bcf-a6fe-b564265224be" />
df.nunique
<img width="1115" height="829" alt="image" src="https://github.com/user-attachments/assets/706bce41-a6ca-4226-829a-330adab6cde1" />
df['Sex'].value_counts()
<img width="1225" height="87" alt="image" src="https://github.com/user-attachments/assets/a8f953c1-3255-4c6f-8d21-602d296e41bf" />
df.Survived.unique()
<img width="1209" height="43" alt="image" src="https://github.com/user-attachments/assets/cb612ca6-96a3-48e0-8b98-f4f0f0d2f9af" />
df.rename(columns={"Sex":"Gender"},inplace=True)

df

<img width="1326" height="515" alt="image" src="https://github.com/user-attachments/assets/7d231c7d-8bf3-401c-acee-3f16c89a2ffd" />
import seaborn as sns

sns.countplot(data=df)

<img width="1108" height="588" alt="image" src="https://github.com/user-attachments/assets/27b4aadd-c507-4902-8938-ba3e1fca6746" />
sns.countplot(x="Survived",hue="Gender",data=df)

<img width="1110" height="572" alt="image" src="https://github.com/user-attachments/assets/fef4c29f-9c3b-4ee7-b64b-c3aa2be8cc9a" />
sns.catplot(x="Survived",hue="Gender",data=df,kind="count")

<img width="1131" height="643" alt="image" src="https://github.com/user-attachments/assets/7f75b7c0-3877-4949-99e3-5509c3e3ddc1" />
sns.catplot(x="Survived",hue="Gender",data=df,kind="violin")

<img width="1132" height="637" alt="image" src="https://github.com/user-attachments/assets/235a3625-dee3-4f7c-b365-ccaeca3db238" />
sns.boxplot(data=df)
<img width="1264" height="549" alt="image" src="https://github.com/user-attachments/assets/0aeeb809-ae81-4d73-b022-b251bd431630" />
df.boxplot(column="Survived",by="Gender")

<img width="1275" height="594" alt="image" src="https://github.com/user-attachments/assets/999b2b49-2e03-4e97-88a4-3c76b16b55e1" />
sns.scatterplot(data=df)

<img width="962" height="545" alt="image" src="https://github.com/user-attachments/assets/9499ba82-8a50-4931-853b-ff9e105da423" />
sns.scatterplot(x=df['Age'],y=df['Fare'])

<img width="1267" height="552" alt="image" src="https://github.com/user-attachments/assets/3c018682-f416-4fab-98ac-3da0a5b009fb" />
sns.jointplot(x='Age',y='Fare',data=df)

<img width="1256" height="763" alt="image" src="https://github.com/user-attachments/assets/d8816bcd-7f4f-429b-9962-3793a12a7804" />
sns.jointplot(x='Age',y='Fare',data=df,kind="kde")
<img width="1279" height="762" alt="image" src="https://github.com/user-attachments/assets/fc6e3770-787a-4eaf-bfda-08a4a5d534c2" />

sns.jointplot(x='Age',y='Fare',data=df,kind="hist")
<img width="1030" height="759" alt="image" src="https://github.com/user-attachments/assets/6e583508-a7d7-46fc-ad82-d6ad8e5bdfcd" />
sns.catplot(x='Gender',col='Survived',data=df,kind='count',color='green')

<img width="1298" height="644" alt="image" src="https://github.com/user-attachments/assets/d9eb1761-6656-482d-bdff-06f1ac884121" />
sns.pairplot(data=df)

<img width="901" height="927" alt="image" src="https://github.com/user-attachments/assets/f69b1b8c-c9a3-434e-9585-130334286fc2" />
corr1=df.select_dtypes(include=["number"]).corr()

sns.heatmap(corr1,annot=True)

<img width="1066" height="632" alt="image" src="https://github.com/user-attachments/assets/df233535-9b29-48d9-8d02-a660afc686e4" />
sns.catplot(x='Gender',col='Survived',data=df,kind='count',hue="Pclass")

<img width="1353" height="640" alt="image" src="https://github.com/user-attachments/assets/6a1353e3-76c2-43c5-aaf6-47cb304521b2" />
import matplotlib.pyplot as plt

fig,ax1=plt.subplots(figsize=(8,5))

pt=sns.boxplot(ax=ax1,x='Pclass',y='Age',hue='Gender',data=df)

<img width="1159" height="558" alt="image" src="https://github.com/user-attachments/assets/41c3e61f-d2a5-4a79-913f-d298489b364e" />






## RESULT
Thus performing Exploratory Data Analysis on the given data set.


