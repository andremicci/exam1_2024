def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total}, Trainable parameters: {trainable}')
    return None


def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()
