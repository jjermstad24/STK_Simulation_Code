import random
import decimal

def Random_Decimal(t):
    lower,upper = t
    return float(decimal.Decimal(random.randrange(lower*10000,upper*10000))/10000)

def Print_Spacing(len=50):
    for i in range(len):
        print("-",end="")
    print()