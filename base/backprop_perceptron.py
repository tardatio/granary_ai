
def forward(w,s,b,y):
    Yhat= w * s + b
    output = (Yhat-y)**2
    return output, Yhat

def derivative_W(x, output, Yhat, y):
    return ((2 * output) * (Yhat - y)) * x # w

def derivative_B(b, output, Yhat, y):
    return ((2 * output) * (Yhat - y)) * b #bias

def main():
    w = 1.0   #weight
    x = 2.0   #sample
    b = 1.0   #bias
    y = 2.0*x #rule

    learning = 1e-1
    epoch = 3

    for i in range(epoch+1):

        output, Yhat = forward(w,x,b,y)
        print("-----------------------------------------------------------------------------")
        print("w:",w)
        print("\tw*b:",w*x)
        print("x:",x,"\t\tsum:", w*x+b)
        print("\tb:",b,"\t\t\tg1:",abs(Yhat-y),"\tg2:",abs(Yhat-y)**2,"\tloss:",output)
        print("\t\tY=2*x:", y)
        print("-----------------------------------------------------------------------------")

        if output == 0.0:
            break


        gw = derivative_W(x, output, Yhat, y)
        gb = derivative_B(b, output, Yhat, y)
        w -= learning * gw
        b -= learning * gb

if __name__ == '__main__':
    main()
