import matplotlib.pyplot as plt
import seaborn as sns
import data
import model

sns.set()

samples = 100
dim = 1
x, y = data.gen_toy_data(samples, dim)
model = model.Linear_reg()
f = x * b
b = model.find_beta(x, y)
print(model.RSS(x, y))
plt.plot(x, y, "o")
plt.plot(x, f)
plt.grid(["on"])
plt.show()
