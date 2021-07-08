import math
import numpy as np
import random
import matplotlib.pyplot as plt
import optimization_functions as of

def log_like_cauchy(alpha, beta = 0.1):
    data = np.array([-4.2, -2.85, -2.3, -1.02, 0.7, 0.98, 2.72, 3.50])
    n = 8
    return(n*math.log(beta)-n*math.log(math.pi) - sum(np.log(beta**2 + np.square(data - alpha))))

def objective(x, function = "log_like_cauchy"):
    if function == "log_like_cauchy":
        return log_like_cauchy(x)
    if function == "noisy":
        return (x + np.random.randn(1)*0.3)**2.0
    return log_like_cauchy(x)

def get_inputs(x_range, generation_size):
    # return np.linspace(-3, -2, generation_size)
    return np.linspace(x_range[0], x_range[1], generation_size)

class OptimizationParams:
    def __init__(self, name):
        self.name = name
        if name == "log_like_cauchy":
            self.x_range = [-5,5]
            self.y_range = [-42, -32]
            self.find_max = True
            self.x_lab = "Alpha"
            self.y_lab = "Log Like Cauchy"
        elif name == "noisy":
            self.x_range = [-5,5]
            self.y_range = [0, 30]
            self.find_max = False
            self.x_lab = "X"
            self.y_lab = "Noisy Value"
        self.plot_objective(show = False)

    def plot_objective(self, show = True):
        n = 1000
        func = lambda x: objective(x, self.name)
        inputs = np.linspace(self.x_range[0], self.x_range[1], num = n)
        outputs = np.array(list(map(func, inputs)))
        plt.plot(inputs, outputs)
        plt.xlabel(self.x_lab)
        plt.ylabel(self.y_lab)
        plt.ylim(self.y_range)
        plt.xlim(self.x_range)
        if (show):
            plt.show()


def genetic_algorithm_1d(generation_size, max_num_generations = 1000, mutation_prob = 0.1,
                            early_stopping = 1000, objective_func = "log_like_cauchy"):
    opt = OptimizationParams(objective_func)

    top_half_index = math.ceil(generation_size/2)
    print("top half index: {}".format(top_half_index))
    best_input = 0
    best_val = 1000000
    if opt.find_max:
        best_val *= -1
    last_update = 2
    prop_cross_selection = of.get_prob_cross_selection(top_half_index)
    # Set Initial Values
    inputs = get_inputs(opt.x_range, generation_size)
    print("Initial Values")
    print(inputs)
    obj_func = lambda i_var: objective(i_var, objective_func)

    for i in range(2,max_num_generations):
        fitness = np.array(list(map(obj_func, inputs)))
        inputs = [x for _, x in sorted(zip(fitness, inputs), reverse = opt.find_max)]

        # Find Best Alpha
        if opt.find_max:
            generation_best_val = max(fitness)
        else:
            generation_best_val = min(fitness)
        if  (generation_best_val < best_val and not opt.find_max) or (generation_best_val > best_val and opt.find_max):
            best_input = inputs[0]
            best_val = generation_best_val
            last_update = i
        # Early Stopping
        if ( (i-last_update) > early_stopping):
            print("early_stopping at {}".format(i))
            return(best_input)
        # Crossover
        for j in range(top_half_index, generation_size,2):
            index_samples = np.random.choice(list(range(top_half_index)), size = 2,
                                            p = prop_cross_selection)
            b = np.random.uniform()
            replace = inputs[j]
            inputs[j] = b*inputs[index_samples[0]] + (1-b)*inputs[index_samples[1]]
            if j+1 < generation_size:
                replace = inputs[j+1]
                inputs[j+1] = b*inputs[index_samples[1]] + (1-b)*inputs[index_samples[0]]
        # Mutations
        for j in range(generation_size):
            if np.random.random() < mutation_prob:
                tau = random.randint(-1, 1)
                r = np.random.uniform()
                inputs[j] += tau*(opt.x_range[1] - opt.x_range[0]) \
                            *(1 - r**((1-i/max_num_generations)**2))
                # inputs[j] = np.random.normal(loc = inputs[j], scale = 0.5)
            inputs[j] = of.limit_range(inputs[j], opt.x_range)


        # Plot guesses
        y_values = np.full(shape = generation_size,
                            fill_value = (opt.y_range[1] - opt.y_range[0]) \
                                        *i/max_num_generations + opt.y_range[0])
        plt.scatter(inputs, y_values, c = 'k', s=10, alpha = 0.5)
    return(best_input)



np.random.seed(1)
func_name = "noisy"
func_name = "log_like_cauchy"
best_alpha = genetic_algorithm_1d(generation_size = 4,
                                max_num_generations = 100,
                                early_stopping = 200,
                                mutation_prob = 0.1,
                                objective_func = func_name)
print(best_alpha)
plt.axvline(x=best_alpha, c="r")
plt.show()
