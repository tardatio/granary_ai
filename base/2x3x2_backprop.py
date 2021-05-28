# based on above website
# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

import numpy as np

def softmax(x,w1,b1,w2,b2):
    # step one
    z = 1/(1+np.exp(-x.dot(+w1)-b1))
    # step two
    output = 1/(1+np.exp(-z.dot(+w2)-b2))
    return output, z

def derivative(output,z,w2,x):
    # derivative of logistic function
    w2_ravel = np.ravel(w2)

    part1 = np.ravel(1-output)
    part2 = np.ravel(output*part1)
    part3 = np.ravel(1* z)
    w2_part = part1*part2

    # derivative w2
    mu3 = np.mean(part3)
    g2_ = w2_part*mu3

    g2 = derivative_transpose(g2_,len(w2))

    #derivative w1
    to_w1 = []
    for i in range(len(w2_ravel)):
        plus = part1*w2_part * w2_ravel[i]
        to_w1.append(np.mean(plus))

    to_w1_1 = part3*(1-part3)
    g1 = []
    for to in to_w1:
        into = np.mean(to*(to_w1_1*i))
        g1.append(into)

    return g2, g1

def derivative_transpose(data, w):
    x = np.array([data])
    dt = np.vstack(list(x)*w).T.ravel()
    return dt

def main():

    x = np.array([[0.05,0.10]])

    # 2 * 3
    w1 = np.array([[0.15,0.20,0.25],[0.20,0.25,0.40]])
    b1 = np.array([0.35])

    # 3 * 2
    w2 = np.array([[0.40, 0.45],[0.50, 0.35],[0.40,0.45]])
    b2 = np.array([0.55])

    T = np.array([0.34,0.66])

    epoch = 5000
    for i in range(epoch+1):
        output,z = softmax(x,w1,b1,w2,b2)
        t = np.square((output-T)) / 2
        total = np.sum(t)

        if i % 100 == 0:
            print(i,": error", total)
            print("--------------------")

        g2,g1 = derivative(output,z,w2,x)

        # output layer
        wp2 = np.ravel(w2) #front
        #wb2 = np.flipud(w2).ravel() # back
        wb2 = np.var(wb2)

        #wb2 = np.var(w2)
        outlayer = []
        for i in range(len(wp2)):
            outlayer.append(wp2[i] - wb2*g2[i])

        # hidden layer
        wp1 = np.ravel(w1) #front
        bx = derivative_transpose(x, len(w1[1])) # back
        hiddenlayer = []
        for i in range(len(wp1)):
            hiddenlayer.append(wp1[i] - bx[i]*g1[i])

        w2 = np.array(outlayer).reshape(len(w2),len(w2.T))
        w1 = np.array(hiddenlayer).reshape(len(w1),len(w1.T))


if __name__ == '__main__':
    main()
