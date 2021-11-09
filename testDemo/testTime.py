# 引入一个time模块， * 表示time模块的所有功能，
# 作用： 可以统计程序运行的时间
from time import *
begin_time = time()
i=0
while i<10000:
    print(i)
    i+=1

end_time = time()
run_time = end_time-begin_time
print ('begin_time：',begin_time) #该循环程序运行时间： 1.4201874732
print ('end_time：',end_time) #该循环程序运行时间： 1.4201874732
print ('该循环程序运行时间：',run_time) #该循环程序运行时间： 1.4201874732

print ('time()：',time()) #该循环程序运行时间： 1.4201874732