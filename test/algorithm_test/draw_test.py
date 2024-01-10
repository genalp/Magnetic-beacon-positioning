import matplotlib.pyplot as plt
import numpy as np

# Assuming the data is read from a file, the following function can be used to process it
def process_data(file_path):
    # Read the data from the file
    with open(file_path, 'r') as file:
        data = np.array([list(map(float, row.split(','))) for row in file])

    # Separate the data into three arrays
    x, y, z = data.T

    # Calculate the Euclidean norm (2-norm) of the vectors
    norms = np.linalg.norm(data, axis=1)

    return x, y, z, norms

# Example usage
Hdata_file_path = 'C:\code\code\Magnetic-beacon-positioning\project\Hdata.txt'
Hx, Hy, Hz, Hnorms = process_data(Hdata_file_path)
Filterdata_file_path = 'C:\code\code\Magnetic-beacon-positioning\project\Filterdata.txt'
Fx, Fy, Fz, Fnorms = process_data(Filterdata_file_path)

# Plotting the three series
# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(12, 6))

# First subplot for the three series
# plt.subplot(2, 1, 1)
plt.plot(Hy[200:350], label='H')
plt.plot(Fy[200:350], label='F')
# plt.plot(z, label='Z')
# plt.title('Three Data Series')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

# # Second subplot for the norms
# plt.subplot(2, 1, 2)
# plt.plot(norms, label='2-Norm')
# plt.title('Euclidean Norm of Vectors')
# plt.xlabel('Index')
# plt.ylabel('Norm Value')
# plt.legend()

# plt.tight_layout()
plt.show()
