"""Command line: `liveness-train --device_ids 0 --patch_info 2.7_80x80` trains one model for one patch type."""
import argparse
import os


def main(argv=None):
    parser = argparse.ArgumentParser(prog="liveness-train", description="Train one MiniFASNet model")
    parser.add_argument("--device_ids", type=str, default="1", help="which gpu id, 0123")
    parser.add_argument("--patch_info", type=str, default="1_80x80",
                        help="[org_1_80x60 / 1_80x80 / 2.7_80x80 / 4_80x80]")
    args = parser.parse_args(argv)

    cuda_devices = [int(elem) for elem in args.device_ids]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cuda_devices))  # before torch touches CUDA
    args.devices = list(range(len(cuda_devices)))

    from training.config import get_default_config, update_config
    from training.trainer import TrainMain

    TrainMain(update_config(args, get_default_config())).train_model()
