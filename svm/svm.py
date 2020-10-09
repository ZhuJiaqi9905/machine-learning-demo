import numpy as np
import math
import random

eps = 1e-6
def read_data(file_name):
    x = []
    y = []
    with open(file_name) as f:
        for line in f:
            tmp = list(map(float, line.split()))
            x.append(tmp[0: 3])
            y.append(1 if(math.isclose(tmp[3], 1)) else -1)
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    return x, y
class Svm:
    def __init__(self, x, y, C, tol, sigma, max_iter):
        self.x = x
        self.y = y
        self.C = C
        self.tol = tol 
        self.sigma = sigma
        self.n= x.shape[0]
        self.m = x.shape[1]
        self.K = np.zeros((self.n, self.n))
        self.alphas = np.zeros((self.n))
        self.b = 0.0
        self.max_iter = max_iter
        self.E = np.zeros((self.n, 2))
    #计算rbf核矩阵
    def rbf_func(self):
        n, m = x.shape
        for i in range(n):
            self.K[i] = np.exp(-np.sum(np.square(self.x - self.x[i]), axis=1) / (2 * self.sigma * self.sigma))
    def smo(self):
        alphas_changed = 0
        iter = 0
        entire_set = True 
        while iter < self.max_iter and ((alphas_changed > 0) or entire_set):
            alphas_changed = 0
            if entire_set: #如果要遍历整个训练集
                for i in range(self.n):
                    alphas_changed += self.select_alphas(i)
            else:
                non_bound = np.nonzero((self.alphas > 0) * (self.alphas < self.C))[0] 
                for i in non_bound:
                    alphas_changed += self.select_alphas(i)
            iter += 1
            if entire_set: 
                entire_set = False
            elif alphas_changed == 0:
                entire_set = True
        
        return
    def get_E(self, i):
        return np.sum(self.alphas * self.y * self.K[:, i]) + self.b - self.y[i]
    def change_alpha(self, i, j):
        #同时改变第a个alpha和第b个alpha
        alpha1_old = self.alphas[i]
        alpha2_old = self.alphas[j]
        y1 = self.y[i]
        y2 = self.y[j]
        if math.isclose(y1, y2):
            #如果y1 == y2
            L = max(0, alpha1_old + alpha2_old - self.C)
            H = min(self.C, alpha1_old + alpha2_old)
        else:
            L = max(0, alpha2_old - alpha1_old)
            H = min(self.C, self.C + alpha2_old - alpha1_old)
        #E1 = np.sum(self.alphas * self.y * self.K[:, i]) + self.b - y1
       # E2 = np.sum(self.alphas * self.y * self.K[:, j]) + self.b - y2
        E1 = self.get_E(i) 
        E2 = self.get_E(j)
        eta = self.K[i, i] + self.K[j, j] - 2*self.K[i, j]
        alpha2_newUnc = alpha1_old + y2 * (E1 - E2) / eta 
        #计算alpha2_new
        if alpha2_newUnc > H:
            alpha2_new = H
        elif alpha2_newUnc < L:
            alpha2_new = L 
        else:
            alpha2_new = alpha2_newUnc
        #计算alpha1_new
        alpha1_new = alpha1_old + y1 * y2 * (alpha2_old - alpha2_new)
        #更新alpha
        self.alphas[i] = alpha1_new
        self.alphas[j] = alpha2_new
        #在更新alpha后，要更新b和E
        self.E[i] = [1, self.get_E(i)]
        self.E[j] = [1, self.get_E(j)]
        b1 = -E1 - y1*self.K[i, i]*(alpha1_new - alpha1_old) - y2*self.K[j, i]*(alpha2_new - alpha2_old) + self.b 
        b2 = -E2 - y1*self.K[i, j]*(alpha1_new - alpha1_old) - y2*self.K[j, j]*(alpha2_new - alpha2_old) + self.b 
        if 0 < self.alphas[i] < self.C:
            self.b = b1 
        elif 0 < self.alphas[j] < self.C:
            self.b = b2 
        else:
            self.b = (b1 + b2) / 2 
        
    def train(self):
        self.rbf_func()
        self.smo() 
        for j in range(self.n):
            if self.alphas[j] > 0 and self.alphas[j] < self.C:
                #计算参数b
                self.b = self.y[j] - np.sum(self.alphas * self.y * self.K[ :,j])
                break
    #根据训练得到的参数，进行决策
    #x是测试数据，二维数组
    def decision(self,x):
        #首先生成x中每个数据和训练集每个数据的核函数
        ker = np.zeros((x.shape[0], self.n))
        for i in range(x.shape[0]):
            ker[i] = np.exp(-np.sum(np.square(self.x - x[i]), axis=1) / (2 * self.sigma * self.sigma))
        return np.sign(np.dot(ker, self.alphas * self.y) + self.b)
    
    def select_alphas(self, i):
        g_i = (np.sum(self.alphas * self.y * self.K[i, :]) + self.b )
        f = g_i * self.y[i]
        Ei = g_i - self.y[i]
        if (math.isclose(self.alphas[i], 0) and f <= 1) or (0 < self.alphas[i] < self.C and (not math.isclose(f, 1)) ) \
            or (math.isclose(self.alphas[i], self.C) and f > 1) :
            #如果违背了KKT条件, 就选择这个alpha_i
            j = self.select_alpha2(i, Ei) #选择alpha_j
            self.change_alpha(i, j)
            return 1
        return 0
    def select_alpha2(self, i, Ei):
        maxK = -1
        max_deltaE = 0
        self.E[i] = [1, Ei]
        choice_list = np.nonzero(self.E[:, 0])[0]
        if(len(choice_list) <= 1):
            #随机选一个
            j = self.random_select(i)
            return j
        else:
            for k in choice_list:
                if k == i: continue
                Ek = (np.sum(self.alphas * self.y * self.K[k, :]) + self.b ) - self.y[k]
                deltaE = abs(Ei - Ek)
                #寻找最大的abs(Ei - Ek)
                if(deltaE > max_deltaE):
                    maxK = k
                    max_deltaE = deltaE
            return maxK 
    def random_select(self, i):
        j = i
        while i == j:
            j = int(random.uniform(0, self.m))
        return j
#x = np.array([1, 2, 3, 4, 5, 6]).reshape((3, 2))
#y = np.array([0, 0, 0, 1, 1, 1])

x, y = read_data("data.txt")
svm = Svm(x, y, 1, 1e-3, 0.1, 100)
svm.train()
y_hat = svm.decision(x) 
correct_accu = np.sum(y_hat == y) / x.shape[0]
print(correct_accu)
 