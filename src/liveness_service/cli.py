"""Command line: `liveness-service predict IMAGE`. Training has its own command, `liveness-train`."""
import argparse
import sys


def predict(args):
    import cv2
    from liveness_service.ml.checker import LivenessChecker

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


def main(argv=None):
    parser = argparse.ArgumentParser(prog="liveness-service", description="Passive face liveness check")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="check whether the face in an image is real")
    p.add_argument("image", help="path to the image")
    p.add_argument("--model_dir", help="folder of .pth models (default: resources/anti_spoof_models)")
    p.add_argument("--device_id", type=int, default=0, help="which gpu id, [0/1/2/3]")
    p.add_argument("--output", help="save a copy of the image with the box and verdict drawn on it")
    p.set_defaults(func=predict)

    args = parser.parse_args(argv)
    args.func(args)
