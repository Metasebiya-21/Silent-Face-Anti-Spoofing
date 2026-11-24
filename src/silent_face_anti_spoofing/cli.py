"""Command line: `silent-face-anti-spoofing predict IMAGE` and `silent-face-anti-spoofing train`."""
import argparse
import os
import sys


def predict(args):
    import cv2
    from silent_face_anti_spoofing.checker import LivenessChecker

    image = cv2.imread(args.image)
    if image is None:
        sys.exit(f"cannot read image: {args.image}")
    kwargs = {"model_dir": args.model_dir} if args.model_dir else {}
    result = LivenessChecker(device_id=args.device_id, **kwargs).check(image)
    if result is None:
        print(f"No face found in '{args.image}'.")
        return
    print(f"'{args.image}' is {'Real' if result.is_real else 'Fake'} Face. Score: {result.score:.2f}.")

    if args.output:  # draw the box and verdict, as the original test.py did
        x, y, w, h = result.bbox
        color = (255, 0, 0) if result.is_real else (0, 0, 255)
        text = f"{'RealFace' if result.is_real else 'FakeFace'} Score: {result.score:.2f}"
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5 * image.shape[0] / 1024, color)
        cv2.imwrite(args.output, image)


def train(args):
    cuda_devices = [int(elem) for elem in args.device_ids]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cuda_devices))
    args.devices = list(range(len(cuda_devices)))

    # imported here so `predict` doesn't need the training-only dependencies
    from silent_face_anti_spoofing.training.config import get_default_config, update_config
    from silent_face_anti_spoofing.training.trainer import TrainMain

    TrainMain(update_config(args, get_default_config())).train_model()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="silent-face-anti-spoofing", description="Silent face anti-spoofing")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="check whether the face in an image is real")
    p.add_argument("image", help="path to the image")
    p.add_argument("--model_dir", help="folder of .pth models (default: resources/anti_spoof_models)")
    p.add_argument("--device_id", type=int, default=0, help="which gpu id, [0/1/2/3]")
    p.add_argument("--output", help="save a copy of the image with the box and verdict drawn on it")
    p.set_defaults(func=predict)

    t = sub.add_parser("train", help="train one model for one patch type")
    t.add_argument("--device_ids", type=str, default="1", help="which gpu id, 0123")
    t.add_argument("--patch_info", type=str, default="1_80x80",
                   help="[org_1_80x60 / 1_80x80 / 2.7_80x80 / 4_80x80]")
    t.set_defaults(func=train)

    args = parser.parse_args(argv)
    args.func(args)
