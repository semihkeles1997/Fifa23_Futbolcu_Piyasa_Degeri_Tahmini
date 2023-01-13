# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 19:53:17 2022

@author: semih
"""

# Kütüphaneler
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile
import AI_KNN_Algorithm as knnAlgorithm
import AI_Decision_Tree_Algorithm as dtreeAlgorithm
import AI_Linear_Regression_Algorithm as lrAlgorithm
import matplotlib.pyplot as plt
import AI_Support_Vector_Machine_Algorithm as svmAlgorithm
import AI_Neural_Network_Algorithm as neuralAlgorithm
# -----------------------------------------------------------------------------


dataset_original = pd.read_csv("fifa23_players_with_club_overall.csv")
X = dataset_original.copy()  # Gerçek datasetin klonunu oluşturup onu kullanıyoruz.

print("-------------------------------")

# Kategorik Verilerin İstenen Numerik Değerleri Dönüştürülmesi
categorical_to_nums = {'AttackingWorkRate'  :   {'Low'  : 0, 'Medium' : 1, 'High' : 2},
                       'DefensiveWorkRate'  :   {'Low'  : 0, 'Medium' : 1, 'High' : 2}} 

# -----------------------------------------------------------------------------
X = X.fillna(0)
#X = X.replace('Not in team',0)
X = X.replace(categorical_to_nums)
y = dataset_original['ValueEUR'] * 0.001  # y değerleri çok büyük olduğundan küçültüyorum.
original_y = dataset_original['ValueEUR']


# Functions
# Burada kontrat süresinden günümüz yılını çıkarıp kontratın bitmesine kalan süreyi buluyoruz. Eğer futbolcu serbest statüde ise 0 olarak alıyoruz.
def Remaining_Contract(contract_until, club_joined):
    remaining = 0 if contract_until-club_joined < 0 else contract_until-club_joined
    return remaining


# -----------------------------------------------------------------------------
"""
Burada da oyuncunun kontratının bitmesine kalan süreyi bu yılla çıkarıp buluyoruz.
"""
X['Remaining_Contract'] = X.apply(lambda x: 
                                    Remaining_Contract(x.ContractUntil, 
                                                       datetime.now().year),axis=1)
# -----------------------------------------------------------------------------


# Target Encoders : Özniteliklerin ağırlıklarını hesaplıyorum.
"""
Burada da oyuncunun hangi ülkenin vatandaşı olduğundan ziyade oynadığı milli takımın piyasa değerine etkisine bakıyoruz. Bu sayede oyuncu ağırlığı yüksek olan bir ülkenin vatandaşı ise ancak o ülkenin milli takımında oynamıyorsa ağırlığı daha düşük oluyor. Sadece ülkesine baksaydık milli takımda oynayanlar ile oynamayanlar aynı ağırlığa sahip olacaklardı.

Aynı şekilde futbolcunun tercih edilen ayağının, kulüpte oynadığı pozisyonun, oynayabildiği pozisyonların, en iyi olduğu pozisyonun, oynadığı kulübün ayrı ayrı piyasa değerine göre ağırlıklarını hesaplıyoruz. Bunlara LabelEncoder ya da OneHotEncoder uygulayamazdım çünkü LabelEncoder uygularsam hangi özelliğe neye göre ağırlık verebileceğimi bilmiyorum. OneHotEncoder da uygulayamazdım çünkü bu özelliklerin piyasa değerine etkisinin olmadığını bilmiyorum. Dolayısıyla ağırlıkları bulmam gerekiyor. 

Bunu yaparken başta TargetEncoder fonksiyonunu kullandım ancak burada Qatar takımına ait 53 futbolcunun hiçbirinin piyasa değeri olmadığından bu değerler için ağırlık değeri çok farklı çıktı. Ben de aşağıdaki yolu kullandım. TargetEncoder ile aynı mantıkta çalışıyor. Test ettim değerler aynı. Tek fark Qatar takımı için bulunan 8.754456-e gibi değerin 0 çıkması. Ki benim de istediğim buydu. 

Daha sonra bulunan değerler piyasa değerlerinin çok büyük sayılar olması sebebiyle aşırı büyük çıktı. Ben de tüm değerleri 0.001 katsayı ile çarparak değerleri nispeten küçültmüş oldum. Tüm değerler için bu çarpımı uyguladığımdan herhangi bir fark olmayacaktır.
"""

X["National_Team_Weight"] = X.groupby("NationalTeam")["ValueEUR"].transform("mean") * 0.000001

X["Preferred_Foot_Weight"] = X.groupby("PreferredFoot")["ValueEUR"].transform("mean") * 0.000001

X["Club_Position_Weight"] = X.groupby("ClubPosition")["ValueEUR"].transform("mean") * 0.000001

X["Positions_Weight"] = X.groupby("Positions")["ValueEUR"].transform("mean") * 0.000001

X["Best_Position_Weight"] = X.groupby("Positions")["ValueEUR"].transform("mean") * 0.000001

X["Club_Name_Weight"] = X.groupby("Club")["ValueEUR"].transform("mean") * 0.000001


X = pd.concat([X,pd.get_dummies(dataset_original['OnLoad'], prefix='OnLoad')],axis=1)
# -----------------------------------------------------------------------------


"""
Datasetten gereksiz ya da düzenlediğimiz verileri çıkarıyoruz.
"""
X = X.drop(['NationalNumber', 'ID', 'Name','FullName',
              'PhotoUrl','ValueEUR','WageEUR','ReleaseClause',
              'ClubNumber','ContractUntil','ClubJoined',
              'NationalTeam','Nationality','Positions',
              'BestPosition','Club','ClubPosition',
              'NationalPosition','PreferredFoot','OnLoad'],axis=1)




mutual_info = mutual_info_regression(X,y)
mutual_info = pd.Series(mutual_info, index=X.columns)
print(mutual_info)


print(mutual_info.sort_values(ascending=False))

plt.title("Özniteliklerin Ağırlık Sıralaması")
plt.xlabel("Öznitelikler")
plt.ylabel("Ağırlık Değerleri")
mutual_info.sort_values(ascending=False).plot.bar(figsize=(15,5))
plt.show()




# Train & Test Split
x_norm = (X-np.min(X))/(np.max(X)-np.min(X))

X_train, X_test, y_train, y_test = train_test_split(x_norm, y, test_size=0.2, random_state=0)


# En iyi öz niteliklerin seçilmesi
selected_top_columns = SelectPercentile(mutual_info_regression, percentile=20)
selected_top_columns.fit(X_train,y_train)
print("------------------")
print(selected_top_columns.get_support())
print("------------------")
print(X_train.columns[selected_top_columns.get_support()])

# En iyi özniteliklerden oluşan yeni bir dataframe oluşturulması ve train, test bölümlenmesi
newdf = x_norm[selected_top_columns.get_feature_names_out()]
X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(newdf,y)

allcolumns = X.columns
aaaa_str_dataset_all_columns = ""
for ex in dataset_original.columns:
    aaaa_str_dataset_all_columns += " '{0}', ".format(ex)


# MODELLER

"""
# KNN
knn_values = knnAlgorithm.KNN_Algorithm_Semih(X_new_train, X_new_test, y_new_train, y_new_test)

# Decision Tree
decision_tree_values = dtreeAlgorithm.Decision_Tree_Semih(X_new_train, X_new_test, y_new_train, y_new_test)

# Support Vector Machine
svm_values = svmAlgorithm.Support_Vector_Machine_Semih(X_new_train, X_new_test, y_new_train, y_new_test)

# Linear Regression
linear_regression_values = lrAlgorithm.Linear_Regression_Semih(X_new_train, X_new_test, y_new_train, y_new_test)

# Neural Network 
neural_values = neuralAlgorithm.Nural_Network_Semih(X_new_train, X_new_test, y_new_train, y_new_test)
"""

# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------


# Comment Lines

"""
Buradaki kodlar ile teams_fifa23.csv dosyasındaki takım isimleri ile
players_fifa23.csv dosyasındaki takımları karşılaştırıp bu takımların
overall değerlerini ekledik. Bu sayede takımların güçlerini oranlayabileceğiz.

dataset_with_overall = dataset_original.copy()
dataset_with_overall['Club_Overall'] = 0
teams_dataset = pd.read_csv("teams_fifa23.csv")
X_sayac = 0
teams_sayac = 0

for i in X.Club:
    for x in teams_dataset.Name:
        if x == i:
            dataset_with_overall.Club_Overall[X_sayac] = teams_dataset.Overall[teams_sayac]
        teams_sayac += 1
    teams_sayac = 0
    X_sayac += 1
    
dataset_with_overall.to_csv('fifa23_players_with_club_overall.csv', index=False)



Bu yolla koşullu ifade yazabiliyoruz.
X['HEYYO'] = X['NationalTeam'].apply(lambda x: 1 if x != 'Not in team'  else 0)



# MinMaxScaler
X_train['Preferred_Foot_Weight'] = MinMaxScaler().fit_transform(ce.TargetEncoder().fit_transform(X_train['Preferred_Foot_Weight'],y))
X_train['Club_Position_Weight'] = MinMaxScaler().fit_transform(ce.TargetEncoder().fit_transform(X_train['Club_Position_Weight'],y))
X_train['Positions_Weight'] = MinMaxScaler().fit_transform(ce.TargetEncoder().fit_transform(X_train['Positions_Weight'],y))
X_train['Best_Position_Weight'] = MinMaxScaler().fit_transform(ce.TargetEncoder().fit_transform(X_train['Best_Position_Weight'],y))
#X_train['OnLoad_Weight'] = MinMaxScaler().fit_transform(ce.TargetEncoder().fit_transform(X_train['OnLoad'],y))




 # Target Encoder
X['Preferred_Foot_Weight'] = ce.TargetEncoder().fit_transform(X['PreferredFoot'],y)
X['Club_Position_Weight'] = ce.TargetEncoder().fit_transform(X['ClubPosition'],y)
X['Positions_Weight'] = ce.TargetEncoder().fit_transform(X['Positions'],y)
X['Best_Position_Weight'] = ce.TargetEncoder().fit_transform(X['BestPosition'],y)

X['Club_Name_Weight'] = ce.TargetEncoder().fit_transform(X['Club'],y)




columns = ['Age','Height','Weight','Overall','Potential',
 'Growth','TotalStats','BaseStats','IntReputation',
 'WeakFoot','SkillMoves','PaceTotal','ShootingTotal','PassingTotal',
 'DribblingTotal','DefendingTotal','PhysicalityTotal','Crossing',
 'Finishing','HeadingAccuracy','ShortPassing','Volleys','Dribbling',
 'Curve','FKAccuracy','LongPassing','BallControl','Acceleration','SprintSpeed',
 'Agility','Reactions','Balance','ShotPower','Jumping','Stamina','Strength',
 'LongShots','Aggression','Interceptions','Positioning','Vision','Penalties',
 'Composure','Marking','StandingTackle','SlidingTackle','GKDiving','GKHandling',
 'GKKicking','GKReflexes','STRating','LWRating','LFRating','CFRating','RFRating',
 'RWRating','CAMRating','LMRating','CMRating','RMRating','LWBRating','CDMRating',
 'RWBRating','LBRating','CBRating','RBRating','GKRating','GKPositioning','Club_Overall']

#X[columns] = MinMaxScaler().fit_transform(X[columns])


"""

# -----------------------------------------------------------------------------




























