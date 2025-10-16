# Autonomous-B-Individual-Task

Tensors & Arrays
A batch of images is stored as a tensor with shape (N, C, H, W) where N is the number of images, C is the number of channels, and H/W are height and width. Grayscale images use C=1 and RGB images use C=3. I verified this by creating example tensors and printing their shapes to confirm the dimensions.

Feedforward Neural Network (FFNN)
I built a small FFNN that flattens the 28×28 image and applies two linear layers with a ReLU in between. The non-linear activation is essential; without it, stacking linear layers would behave like a single linear transform and could not model complex decision boundaries. The matrices and vectors in the network implement y = f(Wx + b), which is where linear algebra appears.

Training Loop & Backpropagation
I trained the FFNN on MNIST for a few epochs using cross-entropy loss and SGD. The loop follows the standard steps: forward pass, loss computation, backward pass to compute gradients, and optimizer step to update weights. The training loss decreased over epochs (see MLP_Loss.png), and the notebook prints the test accuracy to show generalization.

CNN vs FFNN + Filter Visualization
I trained a simple CNN with two conv-ReLU-pool blocks and a final linear layer. Compared with the FFNN, the CNN learned faster and reached higher accuracy on MNIST, which matches expectations for image data. I also visualized the first-layer filters (Conv_filters.png); they look like edge and center-surround detectors that pick up strokes in the digits.

Learning-Rate Sweep
I ran the same CNN with three learning rates. A moderate learning rate (0.05) gave the best progress, a small one (0.005) learned but more slowly, and a large one (0.5) barely improved and was close to unstable. This is visible in the LR plot (Ir_Loss.png) where the curves separate clearly.

Convolution Arithmetic
I used the standard formula out = floor((in + 2p − k)/s) + 1 to predict output sizes and confirmed with a PyTorch conv layer. For example, with k=3, s=2, p=1, a 64×64 input becomes 32×32, which matched the library’s output and validated the calculations.

Backpropagation Gradient Check
For a tiny network, I compared the analytical gradients from autograd to numerical finite-difference estimates by perturbing parameters. The absolute difference was very small (on the order of 1e-4–1e-3), which confirms the backprop implementation and loss setup are correct.
