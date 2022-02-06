import numpy as np
from numpy import random as rd
import random
import matplotlib.pyplot as plt
import matplotlib.axes as a1
import seaborn as sns
from scipy.stats import qmc
import math
import nibabel as nib
import pandas as pd
import itk
import itkwidgets
import statistics
import csv
from ipywidgets import interact, interactive, IntSlider, ToggleButtons
# sns.set_style("darkgrid")
import pylab

df = pd.read_csv('air.csv')
print(df)


df = pd.read_csv ('frtr.csv')

# print(df[4])
# print(df)
year_list = df['iyear'].tolist()
# print(year_list,type(year_list))
death_list = df["fatalities"].tolist()

plt.scatter(year_list,death_list)
plt.xlabel("Year")
plt.ylabel("Number of deaths that year")
plt.title("Scatter plot - france terrorism")
plt.show()


a11 = list(random.random()  for i in range(1000))
a12 = list(random.random()  for i in range(1000))
a1 = [a11,a12]

x = a11
y = a12
z = []
for i in range(1000):
    z.append(math.sin(10*x[i]) * math.cos(10*y[i])) 
plt.tricontourf(x,y,z,levels = 10)
plt.show()
# /home/u1320563/Downloads/NOAA-Temperatures.csv
df = pd.read_csv ('2.2.csv')

# print(df[4])
# print(df)
birth_list = df['births'].tolist()
# print(year_list,type(year_list))
date_list = df["date_of_month"].tolist()
month_list = df['month'].tolist()
day_list = df['day_of_week'].tolist()
print(max(birth_list))
i = birth_list.index(max(birth_list))
print(date_list[i], month_list[i])
k= 0
for i in range(len(birth_list)):
    if day_list[i] == 5 and date_list[i] ==13:
        # print(i)
        k+=1


# print(k)
df = pd.read_csv ('NOAA-Temperatures.csv')

# print(df[4])
# print(df)
value_list = df['Value'].tolist()
# print(year_list,type(year_list))
year_list = df["Year"].tolist()

nvalue_list = list((i*9/5)+ 32 for i in value_list)
plt.bar(year_list, nvalue_list)
plt.xlabel("Year")
plt.ylabel("Degrees F+-")
plt.title("Bar plot - NOAA Temprature anomalies data set")
avgtemp = statistics.mean(value_list)
plt.axhline(y= 32, color = 'green', linestyle = '-',label = 'Mean')
plt.legend()
plt.show()
print(value_list)
defining the number of steps
n = 1000
 
#creating two array for containing x and y coordinate
#of size equals to the number of size and filled up with 0's
x = np.zeros(n)
y = np.zeros(n)
z = np.zeros(n)
x2 = np.zeros(n)
y2 = np.zeros(n)
z2 = np.zeros(n)
x3 = np.zeros(n)
y3 = np.zeros(n)
z3 = np.zeros(n)
# filling the coordinates with random variables
for i in range(1, n):
    val = random.randint(1, 6)
    if val == 1:
        x[i] = x[i - 1] + 1
        y[i] = y[i - 1]
        z[i] = z[i - 1]
    elif val == 2:
        x[i] = x[i - 1] - 1
        y[i] = y[i - 1]
        z[i] = z[i - 1]
    elif val == 3:
        x[i] = x[i - 1]
        y[i] = y[i - 1] + 1
        z[i] = z[i - 1]
    elif val == 4:
        x[i] = x[i - 1]
        y[i] = y[i - 1] - 1
        z[i] = z[i - 1]
    elif val == 5:
        x[i] = x[i - 1]
        y[i] = y[i - 1] 
        z[i] = z[i - 1] + 1
    else :
        x[i] = x[i - 1]
        y[i] = y[i - 1] 
        z[i] = z[i - 1] - 1

for i in range(1, n):
    val = random.randint(1, 6)
    if val == 1:
        x2[i] = x2[i - 1] + 1
        y2[i] = y2[i - 1]
        z2[i] = z2[i - 1]
    elif val == 2:
        x2[i] = x2[i - 1] - 1
        y2[i] = y2[i - 1]
        z2[i] = z2[i - 1]
    elif val == 3:
        x2[i] = x2[i - 1]
        y2[i] = y2[i - 1] + 1
        z2[i] = z2[i - 1]
    elif val == 4:
        x2[i] = x2[i - 1]
        y2[i] = y2[i - 1] - 1
        z2[i] = z2[i - 1]
    elif val == 5:
        x2[i] = x2[i - 1]
        y2[i] = y2[i - 1] 
        z2[i] = z2[i - 1] + 1
    else :
        x2[i] = x2[i - 1]
        y2[i] = y2[i - 1] 
        z2[i] = z2[i - 1] - 1
for i in range(1, n):
    val = random.randint(1, 6)
    if val == 1:
        x3[i] = x3[i - 1] + 1
        y3[i] = y3[i - 1]
        z3[i] = z3[i - 1]
    elif val == 2:
        x3[i] = x3[i - 1] - 1
        y3[i] = y3[i - 1]
        z3[i] = z3[i - 1]
    elif val == 3:
        x3[i] = x3[i - 1]
        y3[i] = y3[i - 1] + 1
        z3[i] = z3[i - 1]
    elif val == 4:
        x3[i] = x3[i - 1]
        y3[i] = y3[i - 1] - 1
        z3[i] = z3[i - 1]
    elif val == 5:
        x3[i] = x3[i - 1]
        y3[i] = y3[i - 1] 
        z3[i] = z3[i - 1] + 1
    else :
        x3[i] = x3[i - 1]
        y3[i] = y3[i - 1] 
        z3[i] = z3[i - 1] - 1
        
     
ax = plt.subplot(1, 1, 1, projection='3d')
ax.plot(x, y, z,'s')
ax.plot(x2,y2,z2,'y')
ax.plot(x3,y3,z3,'b')
# ax.scatter(x[-1], y[-1], z[-1])
ax.set_xlabel('x axis')         
ax.yaxis.set_label_text('y axis')
ax.zaxis.set_label_text('z axis')
plt.title("3D Random walk")
plt.show() 
# plotting stuff:
pylab.title("Random Walk ($n = " + str(n) + "$ steps)")
pylab.plot(x, y,z)
pylab.savefig("rand_walk"+str(n)+".png",bbox_inches="tight",dpi=600)
pylab.show()
img_path = '/home/u1320563/Downloads/T2.nii.gz'
img_obj = nib.load(img_path)
# print(type(img_obj))
img_data = img_obj.get_fdata()
print(type(img_data))

# h,w,d = img_data.shape
# print(h,w,d)
maxval = 255
# i = np.random.randint(0,maxval)
channel = 0
plt.imshow(img_data[:,125,:],cmap="hot")
plt.axis('off')
plt.title("Brain layer slice with dimension 2 - slice 125")
plt.show()
a11 = list(float(format(random.random(),".2f"))  for i in range(1000))
a12 = list(float(format(random.random(),".2f"))  for i in range(1000))
a1 = [a11,a12]

sampler = qmc.LatinHypercube(d=1000)
sample = sampler.random(n=2)
a21 = list(float(format(i,".2f")) for i in sample[0])
a22 = list(float(format(i,".2f")) for i in sample[1])
a2 = [a21,a22]
# print(a2)
x = a2[0]
y = a2[1]

df_ = pd.read_csv('d.csv')
df_ = df_.iloc[1:6]

dimensions = list([ dict(range=(df_['scale'].min(), df_['scale'].max()),tickvals = df_['scale'], ticktext = df_['MMSA'],label='Hospital name', values=df_['scale']),
                    dict(range=(df_['high_risk_per_ICU_bed'].min(),df_['high_risk_per_ICU_bed'].max()),label='High risk per ICU bed', values=df_['high_risk_per_ICU_bed']),
                    dict(range=(df_['high_risk_per_hospital'].min(), df_['high_risk_per_hospital'].max()),label='High risk per hospital', values=df_['high_risk_per_hospital']),
                    dict(range=(df_['icu_beds'].min(), df_['icu_beds'].max()),label='Number of ICU beds', values=df_['icu_beds'])
                  ])

fig = go.Figure(data= go.Parcoords(line = dict(color = df_['scale'], colorscale = 'agsunset'), dimensions = dimensions))
fig.update_layout(title_text="Parallel Coords - Hospitals",width=1200, height=800,margin=dict(l=350, r=60, t=60, b=40))

fig.show()


categories = ['incidents_85_99', 'fatal_accidents_85_99', 'incidents_00_14']
    
label_loc = np.linspace(start=0, stop=1.5 * np.pi, num=len(Air_France))

plt.figure(figsize=(8, 8))
plt.subplot(polar=True)
plt.plot(label_loc, Air_France, label='Air France')
plt.plot(label_loc, American, label='American')
plt.plot(label_loc, All_Nippon_Airways, label='All Nippon Airways')
plt.title('Radar chart - Airlines incidents', size=30)
lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)
plt.legend()
plt.show()

plt.scatter(x, y)
plt.xlabel("x-axis of Latin hypercube sampling array")
plt.ylabel("y-axis of Latin hypercube sampling array")
plt.title("Scatter plot of Latin hypercube sampling array")
plt.show()
Z = np.sin(10*a1[0]) * np.cos(10*a1[1])
plt.contourf(Z, levels=[10])
z = []
for i in range(1000):
    z.append(math.sin(10*a1[0][i])* math.cos(10*a1[1][i]))
print(z)
Z = np.sin(10*a1[0]) * np.cos(10*a1[1])
# [X, Y] = np.meshgrid(a1[0], a1[1])
  
[X, Y] = np.meshgrid(a1[0], a1[1])
  
fig, ax = plt.subplots(1, 1)
  
Z =np.sin(10* X) * np.cos(10* Y) 
# plots contour lines
ax.contour(X, Y, Z)
  
ax.set_title('Contour Plot')
ax.set_xlabel('feature_x')
ax.set_ylabel('feature_y')
  
plt.show()

z.append(math.sin(math.radians(10*a1[0][i]))+ math.cos(math.radians(10*a1[1][i])) )
l = list(float(format(0.01*i,".2f"))  for i in range(100))
# print(l)
l2 = rd.normal(loc=50, scale=15, size=200)
# # print(l2)
# # print(min(l2),max(l2))
# fig = plt.figure(figsize =(10, 7))




#Creating plot
l = list(random.uniform(0,1) for i in range(100))
# print(l)
plt.boxplot(l ,vert= False)
plt.xlabel("Number values")
plt.ylabel("Uniform distribution array") 
# a1.Axes.set_yticklabels("Array 1")
plt.title("Boxplot - Uniform distribution array")
plt.text(12,0.9,'Lower whisker', ha = "center", va = "bottom")
plt.text(37,0.85,'1st quartile', ha = "center", va = "bottom")
plt.text(50,0.85,'Median', ha = "center", va = "bottom")
plt.text(64,0.85,'3rd quartile', ha = "center", va = "bottom")
plt.text(88,0.9,'Upper whisker', ha = "center", va = "bottom")
plt.show()

# # show plot
plt.hist(l2,bins = 20,edgecolor='black')
plt.xlabel("Values - normal distributed")
plt.ylabel("Frequency") 
plt.title("Histogram - normal distributed array")
# plt.ylim([0, 12])

plt.show()
    
    
    
wl = [i*100 for i in l]
file = open("sample.bin", "wb")
nl = list(int(i) for i in wl)
byte_uniform = bytearray(nl)
# print(nl)
file.write(byte_uniform)
file.close()
file = open("sample.bin","rb")
number=list(file.read(100))
nn = list(i for i in number)
nn.sort()
print(max(nn), len(nn),nn)

file.close()
y = np.arange(100)/ float(100)
x = nn
plt.xlabel('Uniform distribution values')
plt.ylabel('Cumulative probability of x values')
  
plt.title('Cumulative distribution function as a line graph')
# plt.xlim([0,100])
plt.plot(x, y)

plt.show()






