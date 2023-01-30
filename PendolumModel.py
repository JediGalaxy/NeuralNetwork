# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 16:04:04 2021

@author: Alexander_Maltsev
"""
import matplotlib.pyplot as plt
import pylab
import math
import numpy as np

class Pendulum:
    def __init__(self):
        #характеристики маятника
        self.m = 1 #суммарная масса (исходный 100)
        self.mb = 0.1 # масса маятника (исходный 50)
        self.g = 9.8 # гравитационная постоянная
        self.Q = 10 # угол наклона (исходный -5)
        self.Q1 = 0 # угловая скорость
        self.Q2 = 0 # угловое ускорение
        self.Qg = 0 # желаемое положение угла (в градусах)
        self.l = 10 # длина маятника (исходный 10)
        self.f = 0 # сила прилагаемая к тележке (исходный 2)
        self.t = 2 # время (исходный 100)
        self.dt = 0.01 # единица времени
        self.b = 0 # переменная цикла
        self.Xg = 0 # желаемая позиция тележки
        self.X2 = 0
        self.X1 = 0
        self.X = 0 # текущая позиция тележки
        self.k = 0
        self.er = 0
        self.fl_show = False
        print('[       Инициализация параметров       ]')
#------------------------------------------------------------------------------
    def reset(self):
        self.Q = np.random.uniform(low=-10, high=10)
        self.Q1 = np.random.uniform(low=-0.05, high=0.05)
        self.Q2 = 0 # угловое ускорение
        self.X2 = 0
        self.X = np.random.uniform(low=-1, high=1)
        self.X1 = np.random.uniform(low=-0.05, high=0.05)
        self.er = 0
        self.b = 0 # переменная цикла
        self.k = 0

        self.yy = [] # массив значений угла маятника
        self.xx = [] # массив значений времени
        self.zz = [] # нулевая ось
        self.XX = [] # массив значений позиции тележки
#        self.O1 = [] # правая граница допустимого значения отклонения
#        self.O2 = [] # левая граница допустимого значения отклонения

        return np.asarray([self.X, self.X1, self.Q, self.Q1])
#        return np.asarray([self.Q, self.Q1])
#------------------------------------------------------------------------------
    def Drop(self, action):
        # В зависимости от значения action (1 или 0), присваиваем значение силы
        # В коде библиотеки gym происходит аналогичное присвоение, однако в
        # примере из книги, параметры системы после вычисления всегда разные


        if action == 1:
            self.f = 10 # Чем меньше сила, тем реже период на графике
        else: self.f = -10
#        elif action == 0:
#            self.f = 10
        self.er0 = self.er
        # Переводим градусы в радианы
        self.QRad=math.radians(self.Q)

        # Q2 - угловое ускорение, Q1 - угловая скорость, Q - угол
        self.Q2 = ((self.m*self.g*math.sin(self.QRad))-math.cos(self.QRad)*(self.f+self.mb*self.l*((self.Q1**2)*math.sin(self.QRad))))/((4/3)*self.m*self.l-self.mb*self.l*(math.cos(self.QRad))**2)
        self.Q1 = self.Q2*self.dt + self.Q1
        self.Q = math.degrees(self.Q1*self.dt + self.QRad)

        # X2 - ускорение тележки, X1 - скорость тележки, X - позиция тележки
        self.X2 = ((self.f+self.mb*self.l*((self.Q1**2)*math.sin(self.QRad-self.Q2*math.cos(self.QRad)))/self.m))
        self.X1 = self.X2*self.dt+self.X1
        self.X = self.X1*self.dt+self.X

        self.k += 1
        # Считаем невязку
        self.er += (0.2*(self.Xg-self.X)**2)+(0.8*(self.Qg-self.Q)**2)
        self.er = self.er/self.k

        # Записываем показатели для графиков
        self.b += self.dt
#        self.b += step
        self.xx.append(self.b)
        self.zz.append(0)
        self.XX.append(self.X)
        self.yy.append(self.Q)
#        self.O1.append(2.5)
#        self.O2.append(-2.5)

        obs = np.asarray([self.X, self.X1, self.Q, self.Q1])
#        obs = np.asarray([self.Q, self.Q1])



        if self.Q>30:
            reward = 1.0
            done = True
        elif self.Q<-30:
            reward = 1.0
            done = True
        elif self.X>20:
            reward = 1.0
            done = True
        elif self.X<-20: # - на - дает +
            reward = 1.0
            done = True
#        elif self.er<self.er0:
#            reward = 1.0
#            done = True
        else:
            reward = 1.0
            done = False

        info = ['ok']
        self.fl_show = True
        return obs, reward, done, info
#------------------------------------------------------------------------------
    def animate(self):
        plt.show()
        print('[          Отрисовка графиков          ]')
        plt.close("all")
        plt.figure('Обратный маятник')
        pylab.ylim(-30, 30)
        plt.ylabel('Угол отклонения')
        plt.xlabel('время')
#            plt.plot(self.xx, self.O1, 'g--')
#            plt.plot(self.xx, self.O2, 'g--')
        plt.plot(self.xx, self.zz, 'green')
        plt.plot(self.xx, self.yy, 'red')
        plt.plot(self.xx, self.XX, 'blue')
        print('Качество управления - ', self.er)
#------------------------------------------------------------------------------
    def new(self):
#        self.Q = np.random.uniform(low=-10, high=10)
#        self.Q1 = np.random.uniform(low=-0.05, high=0.05)
#        self.X = np.random.uniform(low=-1, high=1)
#        self.X1 = np.random.uniform(low=-0.05, high=0.05)

        self.Q = 10
        self.Q1 = 0
        self.X = 0
        self.X1 = 0