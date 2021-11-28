tao=input("输入分位:")
a=[1,2,3,4,5,6,7,8,9,10]
b=0.0
sum=0.0
for j in range(1,11):
    for i in range(1,11):
        if(i>=j):
            b =pow(float(tao) * (i-j),2)
        else:
            b=pow((1-float(tao))*(j-i),2)
        sum=sum+b
    sum=sum/10
    print("%.2f"%sum ,end=' ')
    sum=0.0
# sum=0.0
print('\t')
# tao1=input("输入分位:")
a=[1,2,3,4,5,6,7,8,9,10]
b1=0.0
sum1=0.0
for j in range(1,11):
    for i in range(1,11):
        if(i>=j):
            b = pow((i-j),2)
        else:
            b=pow((j-i),2)
        sum1=sum1+b
    sum1=sum1/10
    print("%.2f"%sum1, end=' ')
    sum1=0.0




