import math
import numpy as np
import random
import matplotlib.pyplot as plt
import optimization_functions as of

def sine_cost_function(x, y):
    return x*np.sin(4*x) + 1.1*y*np.sin(2*y)

class MultiVarOptimization:
    def __init__(self, name):
        self.name = name
        if name == "sine_func":
            self.x_range = [0,10]
            self.y_range = [0, 10]
            self.find_max = False
            self.x_lab = "X"
            self.y_lab = "Y"
            self.obj_func = lambda x, y: sine_cost_function(x, y)
        if self.find_max:
            self.best_val = -1000000
        else:
            self.best_val = 1000000
        self.last_update = 2
    def static_plot(self, show = False):
        x = np.linspace(self.x_range[0], self.x_range[1], 50)
        y = np.linspace(self.y_range[0], self.y_range[1], 40)
        X, Y = np.meshgrid(x, y)
        if self.name == "sine_func":
            Z = self.obj_func(X, Y)
        plt.contour(X, Y, Z, 20, cmap='RdGy', zorder=0);
        plt.colorbar();
        plt.xlabel(self.x_lab)
        plt.ylabel(self.y_lab)
        plt.ylim(self.y_range)
        plt.xlim(self.x_range)
        if show:
            plt.show()
    def initial_values(self, generation_size):
        return np.random.rand(generation_size,2)*10
    def update_best_val(self, generation_best_val, generation_best_input, i):
        if  (generation_best_val < self.best_val and not self.find_max) \
                    or (generation_best_val > self.best_val and self.find_max):
            self.best_input = generation_best_input
            self.best_val = generation_best_val
            self.last_update = i
    def early_stopping(self, i, early_stopping):
        if ( (i-self.last_update) > early_stopping):
            print("early_stopping at {}".format(i))
            return(True)
    def crossover(self, inputs, top_half_index, prop_cross_selection):
        # blended crossover
        for j in range(top_half_index, self.generation_size,2):
            index_samples = np.random.choice(list(range(top_half_index)), size = 2,
                                            p = prop_cross_selection)
            b = np.random.uniform()
            for i in range(2):
                inputs[j, i] = b*inputs[index_samples[0], i] \
                                + (1-b)*inputs[index_samples[1], i]
            if j+1 < self.generation_size:
                for i in range(2):
                    inputs[j+1, i] = b*inputs[index_samples[1], i] \
                                    + (1-b)*inputs[index_samples[0], i]
        return inputs

    def mutations(self, inputs, iteration):
        for j in range(self.generation_size):
            if np.random.random() < self.mutation_prob:
                for i in range(2):
                    tau = random.randint(-1, 1)
                    r = np.random.uniform()
                    inputs[j, i] += tau*(self.x_range[1] - self.x_range[0]) \
                                *(1 - r**((1-iteration/self.max_num_generations)**2))
                # inputs[j] = np.random.normal(loc = inputs[j], scale = 0.5)
                    inputs[j,i] = of.limit_range(inputs[j,i], self.x_range)
        return inputs

    def optimize(self, generation_size, max_num_generations = 1000,
                mutation_prob = 0.1, early_stopping = 200):
        self.generation_size = generation_size
        self.mutation_prob = mutation_prob
        self.max_num_generations = max_num_generations
        top_half_index = math.ceil(generation_size/2)
        prop_cross_selection = of.get_prob_cross_selection(top_half_index)
        inputs = self.initial_values(self.generation_size)
        # print("inputs")
        for i in range(1, self.max_num_generations):
            # print("generation {}".format(i))
            fitness = np.array(list(map(self.obj_func, inputs[:,0], inputs[:,1])))
            # print(list(zip(fitness, inputs)))
            inputs = np.array([x for _, x in sorted(zip(fitness, inputs),
                                                    reverse = self.find_max,
                                                    key=lambda x: x[0])])
            # print(inputs)
            fitness = sorted(fitness, reverse = self.find_max)
            plt.scatter(inputs[:,0], inputs[:,1], cmap='cool',  c = fitness, s=10,
                alpha = 1, zorder=1, vmin=-15, vmax=15)
            self.update_best_val(fitness[0], inputs[0, :], i)
            if (self.early_stopping(i, early_stopping)):
                plt.colorbar()
                return
            inputs = self.crossover(inputs, top_half_index, prop_cross_selection)
            inputs = self.mutations(inputs, i)

            # print("After mutations and crossover")
            # print(inputs)

        plt.colorbar()


opt = MultiVarOptimization("sine_func")
opt.optimize(generation_size = 20, max_num_generations = 2000)
opt.static_plot(True)
# plt.show()
