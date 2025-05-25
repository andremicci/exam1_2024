def output_dim(dim, kernel_size, stride=1, padding=1):
    return (int((dim + 2 * padding - kernel_size) / stride) + 1)