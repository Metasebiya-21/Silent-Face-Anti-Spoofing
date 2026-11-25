"""Model-file helpers used at inference (and by training, which names its checkpoints to match)."""


def get_kernel(height, width):
    """Kernel of MiniFASNet's last conv: the input shrinks 16x through the network (80 -> 5)."""
    return ((height + 15) // 16, (width + 15) // 16)


def parse_model_name(model_name):
    """'2.7_80x80_MiniFASNetV2.pth' -> (80, 80, 'MiniFASNetV2', 2.7). A leading 'org' means no crop (scale None)."""
    info = model_name.split('_')[0:-1]
    h_input, w_input = info[-1].split('x')
    model_type = model_name.split('.pth')[0].split('_')[-1]
    scale = None if info[0] == "org" else float(info[0])
    return int(h_input), int(w_input), model_type, scale
