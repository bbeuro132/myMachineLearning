import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#큰 물고기와 작은 물고기를 판별해주는 프로그램의 작성
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
#단순 리스트 형식의 자료

np.column_stack(([1,2,3], [4,5,6]))
#리스트 컴프리헨션을 쓰는 대신 np.column_stack을 사용 (결과는 동일하다)
fish_data = np.column_stack((fish_length, fish_weight))

#print(fish_data[:5])
#print(np.ones(5)) #1을 다섯개 만드는 문법

fish_target = np.concatenate((np.ones(35), np.zeros(14)))
#print(fish_target[33:38])

def doa():
    return 1,2,3,4,5

'''a,b,c,d,e = doa()
print(a)
print(b)
print(c)
print(d)
print(e)'''


train_input, test_input, train_target, test_target\
    = train_test_split(fish_data, fish_target, random_state=42)

'''print("train_input : ",train_input)
print("train_target : ", train_target)

print("test_input : ",test_input)
print("test_target : ", test_target)'''

#plt.scatter(train_input[:,0], train_input[:,1]) #열의 인덱스가 0인 값을 추출, 열의 인덱스가 1인 값을 추출 (1열, 2열)
#plt.scatter(test_input[:,0], test_input[:,1])
#plt.scatter(25, 150, marker='^')
#plt.xlabel('length')
#plt.ylabel('weight')
#plt.show()

knclf = KNeighborsClassifier()
knclf.fit(train_input, train_target)

prevalues = knclf.predict([[25,150]])
#print(prevalues)


distances, indexes = knclf.kneighbors([[25, 150]]) #거리와 위치를 구한다

#plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
#plt.xlim((0, 1000))
#plt.show()


#표준편차 구하기
mean = np.mean(train_input, axis=0) #훈련데이터 평균값
std = np.std(train_input, axis=0) #표준점수

train_scaled = (train_input - mean) / std #표준점수
'''
    [100,200,300] / 100 = [1,2,3]
    [[100,200], [300,400], [500,600]] / 100 = [[1,2], [3,4], [5,6]]
'''

new = ([25,150] - mean) / std
#print(new[0], new[1])


plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1])
plt.xlabel('length')
plt.ylabel('weight')
#plt.show()

knclf = KNeighborsClassifier() #이웃되는 좌표의 기준은 괄호 안에 n_neighbors= 의 숫자를 입력하는걸로 변경한다
knclf.fit(train_scaled, train_target) #train_target는 표준점수로 바꿀 필요가 없음

distances, indexes = knclf.kneighbors([new])

plt.scatter(train_scaled[indexes,0], train_scaled[indexes,1])
plt.show()

prevalues = knclf.predict([new])
print(prevalues)

'''
    train_test_split -> 셔플을 통해 훈련데이터와 테스트 데이터를 나눈다
    
    (실제 데이터 - 평균값) / 표준점수

    x좌표축의 데이터 범위와 y좌표축의 데이터 범위가 다를 때에는
    데이터 전처리가 필요하다
    표준점수로 구해주어야 한다
    
    train_scaled -> 표준점수가 나온 훈련데이터
    KNeighborsClassifier 학습기에 학습을 시켜서
    nebores() 제일 가까운 좌표 다섯개를 구한다.
    5개 데이터를 그래프에 시각화를 한다
'''