import json
import random
random.seed(12412)
with open('dataset_v2.json', 'r') as fp:
        dataset = json.load(fp)
with open('history_item_v2.json', 'r') as fp:
        history_item = json.load(fp)
with open('history_user_v2.json', 'r') as fp:
        history_user = json.load(fp)

with open('history_item_v2.json', 'r') as fp:
        history_item2 = json.load(fp)
with open('history_user_v2.json', 'r') as fp:
        history_user2 = json.load(fp)

train = {"user":[],"item":[],"rate":[]}
val = {"user":[],"item":[],"rate":[]}
test = {"user":[],"item":[],"rate":[]}

train2 = {"user":[],"item":[],"rate":[]}
val2 = {"user":[],"item":[],"rate":[]}
test2 = {"user":[],"item":[],"rate":[]}
def rebuild(a):
    b = {}
    for u,v in a.items():
        b[int(u)] = v
    return b
history_u_lists, history_ur_lists =rebuild(history_user["user_item"]),rebuild(history_user["user_rating"])
history_v_lists, history_vr_lists = rebuild(history_item["item_user"]),rebuild(history_item["item_rating"])
train_u, train_v, train_r = dataset["user"],dataset["item"],dataset["rate"]

history_u_lists2, history_ur_lists2 =rebuild(history_user2["user_item"]),rebuild(history_user2["user_rating"])
history_v_lists2, history_vr_lists2 = rebuild(history_item2["item_user"]),rebuild(history_item2["item_rating"])


train_u, train_v, train_r = dataset["user"],dataset["item"],dataset["rate"]


n_u = len(history_u_lists.keys())
n_i = len(history_v_lists.keys())
#print(history_u_lists2.keys())
for i in range(35705):
    if i not in history_u_lists.keys(): print (i)
print(n_u,n_i)
"""
for i in range(n_u):
    try:
        a = history_u_lists2[i]
    except:
        print("error",i)
for i in range(n_i):
    try:
        a = history_u_lists2[i]
    except:
        print("error",i)
"""
black = []
for item,list_u in history_v_lists.items():
    if len(list_u)<=2:
        black.append(item)

def def_value(): 
    return []
     
from collections import defaultdict    
# Defining the dict 
d = defaultdict(def_value) 
#history_u_lists2, history_ur_lists2 = history_u_lists.copy(), history_ur_lists.copy()
#history_v_lists2, history_vr_lists2 = history_v_lists.copy(), history_vr_lists.copy()

with open("error.txt","w") as writer:
  for i in range(len(train_v)):
    if i % 1000 == 0: print(i)
    if train_v[i] in black:
   
        train["user"].append(train_u[i])
        train["item"].append(train_v[i])
        train["rate"].append(train_r[i])

        train2["user"].append(train_u[i])
        train2["item"].append(train_v[i])
        train2["rate"].append(train_r[i])
        
    else:
        #"""
        if random.random() < 0.2:
            flag = 1
            if len(history_u_lists[int(train_u[i])]) == 1 or len(history_v_lists[int(train_v[i])])==1:
                train2["user"].append(train_u[i])
                train2["item"].append(train_v[i])
                train2["rate"].append(train_r[i]) 
                print("exception")
                flag = 0
                #continue
            if flag :
              try:
                index = history_u_lists[int(train_u[i])].index(train_v[i])
                history_u_lists[int(train_u[i])].remove(train_v[i])
                del history_ur_lists[int(train_u[i])][index]

                index = history_v_lists[int(train_v[i])].index(train_u[i])
                history_v_lists[int(train_v[i])].remove(train_u[i])
                del history_vr_lists[int(train_v[i])][index]
                #print("success")
              except:
                writer.write(str(i)+"\n")
                #print("error")
            
              if random.random() < 0.5:
                val["user"].append(train_u[i])
                val["item"].append(train_v[i])
                val["rate"].append(train_r[i])
              else:
                test["user"].append(train_u[i])
                test["item"].append(train_v[i])
                test["rate"].append(train_r[i])
        else:
            train["user"].append(train_u[i])
            train["item"].append(train_v[i])
            train["rate"].append(train_r[i])
        
        if random.random() < 0.4:
            flag = 1
            if len(history_u_lists2[int(train_u[i])]) == 1 or len(history_v_lists2[int(train_v[i])])==1:
                train2["user"].append(train_u[i])
                train2["item"].append(train_v[i])
                train2["rate"].append(train_r[i]) 
                print("exception2")
                flag = 0
                #continue
            if flag:
              try:
                index = history_u_lists2[int(train_u[i])].index(train_v[i])
                history_u_lists2[int(train_u[i])].remove(train_v[i])
                del history_ur_lists2[int(train_u[i])][index]
                 
                index = history_v_lists2[int(train_v[i])].index(train_u[i])
                history_v_lists2[int(train_v[i])].remove(train_u[i])
                del history_vr_lists2[int(train_v[i])][index]
                #print("success")
              except:
                #pass
                writer.write(str(i)+"\n")
                #print("error")
            
              if random.random() < 0.5:
                val2["user"].append(train_u[i])
                val2["item"].append(train_v[i])
                val2["rate"].append(train_r[i])
              else:
                test2["user"].append(train_u[i])
                test2["item"].append(train_v[i])
                test2["rate"].append(train_r[i])
        else:
            train2["user"].append(train_u[i])
            train2["item"].append(train_v[i])
            train2["rate"].append(train_r[i])        
        
datasets = {"80":{"train":train,"val":val,"test":test},"60":{"train":train2,"val":val2,"test":test2}}

history_user={"user_item":history_u_lists,"user_rating":history_ur_lists}
history_item = {"item_user":history_v_lists,"item_rating":history_vr_lists}
#history_v_lists, history_vr_lists = rebuild(history_item["item_user"]),rebuild(history_item["item_rating"])
history_user2={"user_item":history_u_lists2,"user_rating":history_ur_lists2}
history_item2 = {"item_user":history_v_lists2,"item_rating":history_vr_lists2}
with open('dataset_split.json', 'w') as fp:
    json.dump(datasets, fp) 

with open('history_user_split.json', 'w') as fp:
    json.dump(history_user, fp) 

with open('history_item_split.json', 'w') as fp:
    json.dump(history_item, fp) 

with open('history_user_split2.json', 'w') as fp:
    json.dump(history_user2, fp) 

with open('history_item_split2.json', 'w') as fp:
    json.dump(history_item2, fp) 


