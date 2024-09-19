from scipy.io import loadmat

# Load the .mat file
matrixVar = loadmat("A01T.mat")

# List all variable names in the .mat file to see what it contains
print(matrixVar.keys())

# Now, let's say one of the keys is 'data' (replace with an actual key)
# You can access and see the content of that variable like this:
data = matrixVar['data']

# View a small portion of the data
# print(data)            # View entire array (if not too large)
# print(data.shape)      # View the shape of the data array
# print(data[:5])        # View first 5 rows (replace with actual slicing as needed)

# If it's too large, use slicing or print smaller portions
