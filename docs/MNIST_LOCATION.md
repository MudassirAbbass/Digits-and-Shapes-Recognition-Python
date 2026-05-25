# 📍 MNIST Dataset Location

## Where is MNIST Stored?

The MNIST dataset is **automatically downloaded** by TensorFlow/Keras when you first call `mnist.load_data()`.

### Default Storage Location

The dataset is stored in your user's home directory under `.keras/datasets/`:

#### **Windows:**
```
C:\Users\<YourUsername>\.keras\datasets\mnist.npz
```

#### **Linux/Mac:**
```
~/.keras/datasets/mnist.npz
```

### How to Find It

#### Method 1: Using Python
```python
import os
from pathlib import Path

# Get Keras datasets directory
keras_dir = Path.home() / '.keras' / 'datasets'
print(f"MNIST Location: {keras_dir}")
print(f"Full path: {keras_dir / 'mnist.npz'}")

# Check if it exists
if (keras_dir / 'mnist.npz').exists():
    print("✅ MNIST dataset found!")
    print(f"File size: {(keras_dir / 'mnist.npz').stat().st_size / (1024*1024):.2f} MB")
else:
    print("❌ MNIST not downloaded yet. It will download on first use.")
```

#### Method 2: Check Manually

**Windows:**
1. Open File Explorer
2. Navigate to: `C:\Users\<YourUsername>\.keras\datasets\`
3. Look for `mnist.npz` file

**Linux/Mac:**
```bash
ls -lh ~/.keras/datasets/
```

### How MNIST is Loaded in This Project

In `src/train_models.py`:

```python
from tensorflow.keras.datasets import mnist

# This automatically downloads MNIST if not already downloaded
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

**What happens:**
1. First time: Downloads ~11 MB compressed file from internet
2. Stores in `~/.keras/datasets/mnist.npz`
3. Subsequent times: Loads from cache (no download needed)

### MNIST Dataset Details

- **File Format:** `.npz` (NumPy compressed archive)
- **File Size:** ~11 MB (compressed)
- **Uncompressed:** ~60 MB
- **Contents:**
  - 60,000 training images (28×28 grayscale)
  - 10,000 test images (28×28 grayscale)
  - Labels (0-9 for each image)

### Dataset Structure

```
MNIST Dataset
├── Training Set
│   ├── Images: (60000, 28, 28) - pixel values 0-255
│   └── Labels: (60000,) - digits 0-9
└── Test Set
    ├── Images: (10000, 28, 28) - pixel values 0-255
    └── Labels: (10000,) - digits 0-9
```

### How to Verify MNIST is Downloaded

Run this in Python:
```python
from tensorflow.keras.datasets import mnist
import os

# Try to load MNIST
try:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("✅ MNIST loaded successfully!")
    print(f"Training images: {x_train.shape}")
    print(f"Test images: {x_test.shape}")
    
    # Check file location
    keras_dir = os.path.expanduser('~/.keras/datasets')
    mnist_file = os.path.join(keras_dir, 'mnist.npz')
    if os.path.exists(mnist_file):
        size_mb = os.path.getsize(mnist_file) / (1024 * 1024)
        print(f"\n📁 MNIST file location: {mnist_file}")
        print(f"📦 File size: {size_mb:.2f} MB")
    else:
        print("\n⚠️  MNIST file not found in expected location")
except Exception as e:
    print(f"❌ Error loading MNIST: {e}")
```

### Project-Specific Location

**In this project:**
- MNIST is **NOT** stored in the project folder
- It's stored in the system-wide Keras cache
- The project folder has a `data/` directory, but it's empty (MNIST is not copied there)

### Why Not in Project Folder?

- **Shared across projects:** All TensorFlow/Keras projects use the same cache
- **Saves space:** Don't duplicate 60MB for each project
- **Automatic management:** Keras handles download/caching automatically

### Manual Download (Optional)

If you want to download MNIST manually:

**Original Source:**
- http://yann.lecun.com/exdb/mnist/

**Files:**
- `train-images-idx3-ubyte.gz` (training images)
- `train-labels-idx1-ubyte.gz` (training labels)
- `t10k-images-idx3-ubyte.gz` (test images)
- `t10k-labels-idx1-ubyte.gz` (test labels)

But **you don't need to** - Keras handles it automatically!

---

## Summary

- **Location:** `~/.keras/datasets/mnist.npz` (or `C:\Users\<username>\.keras\datasets\mnist.npz` on Windows)
- **Download:** Automatic on first `mnist.load_data()` call
- **Size:** ~11 MB compressed
- **Access:** Use `mnist.load_data()` - no need to know exact location
