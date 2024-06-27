import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Generate a perfect normal distribution
np.random.seed(0)  # For reproducibility
perfect_normal_data = np.random.normal(loc=0, scale=1, size=1000)

# Generate QQ plot
plt.figure(figsize=(10, 6))
stats.probplot(perfect_normal_data, dist="norm", plot=plt)
plt.title('QQ Plot of Perfectly Normally Distributed Data')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.show()
