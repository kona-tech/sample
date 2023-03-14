import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from IPython.display import HTML
import time



plt.ion()

figure, ax = plt.subplots(figsize=(8,6))

plt.xlabel("X",fontsize=18)
plt.ylabel("Y",fontsize=18)

arr = np.random.randint(0, 2, (100, 100))

for i in range(0,100):
    plt.title("life game?,frame=i")
    next_arr = np.zeros((100,100))
    for x in range(1,arr.shape[0]-1):
        for y in range(1,arr.shape[1]-1):
            alive_num = arr[x-1,y-1] + arr[x,y-1] + arr[x+1,y-1] + arr[x-1,y] + arr[x+1,y]+ arr[x-1,y+1] + arr[x,y+1] + arr[x+1,y+1]
            #誕生
            #死んでいるセルに隣接する生きたセルがちょうど3つあれば、次の世代が誕生する。
            if arr[x,y] == 0 and alive_num == 3:
                  next_arr[x,y] = 1
            
            #生存
            #生きているセルに隣接する生きたセルが2つか3つならば、次の世代でも生存する。
            if arr[x,y] == 1 and alive_num in [2,3]:
                  next_arr[x,y] = 1
            
            #過疎
            #生きているセルに隣接する生きたセルが1つ以下ならば、過疎により死滅する。
            if arr[x,y] == 1 and alive_num <= 1:
                  next_arr[x,y] = 0
                  
            #過密
            #生きているセルに隣接する生きたセルが4つ以上ならば、過密により死滅する。
            if arr[x,y] == 1 and alive_num >= 4:
                  next_arr[x,y] = 0
    
    arr=next_arr
    
    line=plt.imshow(arr)
    figure.canvas.draw()
         
    figure.canvas.flush_events()
    figure.clf
    time.sleep(0.01)
    print(i)    
        
    
