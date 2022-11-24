# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #1 выполнил:
- Батраков Дмитрий Антонович
- НМТ212701
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | # | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
Ознакомиться с основными опреаторами языка Python на примере реализации линейной регрессии.

## Задание 1
Написать программы Hello World на Python и Unity.

- Для Python в отчете приведён скриншот с демонстрацией сохранения
документа google.colab на диск с запуском программы, выводящей
сообщение Hello World.

![image](https://user-images.githubusercontent.com/113825126/203826678-59247928-6b70-4e05-b83a-2b52da3036ef.png)


- Для Unity в отчете приведён скришноты вывода сообщения Hello
World в консоль.

![image](https://user-images.githubusercontent.com/113825126/203826326-9e293125-28af-4e13-b6ff-3e5143449e97.png)

## Задание 2

В разделе «ход работы» пошагово выполнен каждый пункт с описанием и примером реализации задачи по теме лабораторной работы.

Ход работы.
1) Произведена подготовка данных для работы с алгоритмом линейной регрессии.
10 видов данных были установлены случайным образом, и данные находились в линейной зависимости. 
Данные преобразуются в формат массива, чтобы их можно было вычислить напрямую при использовании умножения и сложения.

```py
#Import the required modules, numpy for calculation, and Matplotlib for drawing
import numpy as np
import matplotlib.pyplot as plt
#This code is for jupyter Notebook only
%matplotlib inline
# define data, and change list to array
x = [3,21,22,34,54,34,55,67,89,99]
x = np.array(x)
y = [2,22,24,65,79,82,55,130,150,199]
y = np.array(y)
#Show the effect of a scatter plot
plt.scatter(x,y)
```
2) Определены связанные функции.

Функция модели: определяет модель линейной регрессии wx+b.

Функция потерь: функция потерь среднеквадратичной ошибки.

Функция оптимизации: метод градиентного спуска для нахождения частных производных w и b.

```py
#The basic linear regression model is wx+ b, and since this is a two-dimensional space, the model is ax+ b
def model(a, b, x):
  return a*x + b
  
#Tahe most commonly used loss function of linear regression model is the loss function of mean variance difference
def loss_function(a, b, x, y):
  num = len(x)
  prediction=model(a,b,x)
  return (0.5/num) * (np.square(prediction-y)).sum()
  
#The optimization function mainly USES partial derivatives to update two parameters a and b
def optimize(a,b,x,y):
  num = len(x)
  prediction = model(a,b,x)
  #Update the values of A and B by finding the partial derivatives of the loss function on a and b
  da = (1.0/num) * ((prediction -y)*x).sum()
  db = (1.0/num) * ((prediction -y).sum())
  a = a - Lr*da
  b = b - Lr*db
  return a, b
  
#iterated function, return a and b
def iterate(a,b,x,y,times):
  for i in range(times):
    a,b = optimize(a,b,x,y)
  return a,b
```
3) Начата итерация

Шаг 1 Инициализация и модель итеративной оптимизации

```py
#Initialize parameters and display
a = np.random.rand(1)
b = np.random.rand(1)
Lr = 0.000001
#For the first iteration, the parameter values, losses, and visualization after the iteration are displayed
a,b = iterate(a,b,x,y,1)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```
[0.74555081] [0.15197632] 1775.8115416401874

[<matplotlib.lines.Line2D at 0x7f39a90c67d0>]

![image](https://user-images.githubusercontent.com/113825126/203834179-c91bf42c-d470-4747-bb14-69a10193119c.png)

Шаг 2 На второй итерации отображаются значения параметров, значения потерь и эффекты визуализации после итерации

```py
a,b = iterate(a,b,x,y,2)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```
[0.92032276] [0.48131238] 1260.0865877559056

[<matplotlib.lines.Line2D at 0x7f39a9039f10>]

![image](https://user-images.githubusercontent.com/113825126/203834886-0dd3be5b-f67f-4f04-805a-7ce54f72c41f.png)

Шаг 3 Третья итерация показывает значения параметров, значения потерь и визуализацию после итерации
```py
a,b = iterate(a,b,x,y,3)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```
[0.7521493] [0.67958413] 1731.6096765070213

[<matplotlib.lines.Line2D at 0x7f39a8fbd990>]

![image](https://user-images.githubusercontent.com/113825126/203834655-24392b11-4f50-4a12-b5bc-3099e107e53e.png)

Шаг 4 На четвертой итерации отображаются значения параметров, значения потерь и эффекты визуализации
```py
a,b = iterate(a,b,x,y,4)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```
[1.00278811] [0.40124464] 1059.6117823957447

[<matplotlib.lines.Line2D at 0x7f39a8f411d0>]

![image](https://user-images.githubusercontent.com/113825126/203835118-03de9fc9-9041-4952-8b0e-f5885429e844.png)

Шаг 5 Пятая итерация показывает значение параметра, значение потерь и эффект визуализации после итерации
```py
a,b = iterate(a,b,x,y,5)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```
[0.60927501] [0.71513025] 2206.2570050390327

[<matplotlib.lines.Line2D at 0x7f39a8eb5a50>]

![image](https://user-images.githubusercontent.com/113825126/203835190-7ce0c8a1-318d-4266-840a-d291bc3d9ee4.png)

Шаг 6 10000-я итерация, показывающая значения параметров, потери и визуализацию после итерации
```py
a,b = iterate(a,b,x,y,10000)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```
[1.67100051] [0.55616617] 198.9919676893701

[<matplotlib.lines.Line2D at 0x7f39a8e3e3d0>]

![image](https://user-images.githubusercontent.com/113825126/203835252-61398759-c28c-445f-a8a2-a280147b95ca.png)
## Задание 3 
Изучить код на Python и ответить на вопросы:

- Должна ли величина loss стремиться к нулю при изменении исходных данных? 
Ответить на вопрос, привести пример выполнения кода, который подтверждает ваш ответ.

- Какова роль параметра Lr? Ответить на вопрос, привести пример выполнения кода, который подтверждает ваш ответ. 
В качестве эксперимента можете изменить значение параметра.
