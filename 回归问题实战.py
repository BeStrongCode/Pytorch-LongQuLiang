import numpy as np

# y = wx+b
# loss = (wx+b-y)^2

# 计算损失
def compute_loss(w,b,points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))

# 梯度下降
# w' = w - lr * dloss/dw
# dl/dw = 2(wx+b-y)x
# dl/db = 2(wx+b-y)
def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += (2/N) * (((w_current * x + b_current )) - y) * x
        w_gradient += (2/N) * (w_current * x + b_current - y)
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]
def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    b = starting_b
    w = starting_w
    for i in range(num_iterations):
        b , w = step_gradient(b, w, np.array(points), learning_rate)
    return [b, w]

def generate_data(num_points):
    np.random.seed(42)
    x = 2 * np.random.rand(num_points,1)
    y = 4 + 3 * x + np.random.randn(num_points, 1)
    points = np.hstack((x,y))
    return points

def run():
    # points = np.genfromtxt("data.csv", delimiter=',')
    points = generate_data(100)
    learning_rate = 0.0001
    initial_b = 0
    initial_w = 0
    num_iterations = 100
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b,initial_w,compute_loss(initial_b,initial_w,points)))
    print("Runing....")
    [b , w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, loss = {3}".format(num_iterations, b, w, compute_loss(b,w,points)))

if __name__ == '__main__':
    run()