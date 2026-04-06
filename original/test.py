import scipy.stats as stats
import matplotlib.pyplot as plt

poisson = stats.poisson.pmf(2.5)
plt.plot(poisson)
plt.show()