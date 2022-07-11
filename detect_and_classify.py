import pathlib

from bounding_box_extractor import _build_detection_model, extract_predictions_from_image
from classifier import *
from classifier import _build_classification_model, _classify, _image_to_normalized_tensor

if __name__ == "__main__":

    # some weird warning in PyTorch, does not concern us
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='Detect trash cans, cut them out and classify the cutouts.')

    parser.add_argument('-i', '--input',
                        required=True,
                        nargs='*',
                        help='filepath of the images to apply the model to.')

    parser.add_argument('-o', '--output_mask',
                        default='{filename}_out{index}_{class_name}.{extension}',
                        required=False,
                        help='format string for output file, supports keywords "filename", "extension", "classname" '
                             'and "index". '
                             'default: {filename}_out{index}_{class_name}.{extension}')

    parser.add_argument('-b', '--bounding_box',
                        action='store_true',
                        default=False,
                        help='Include the original bounding box in the cutout.')

    parser.add_argument('-d', '--detection-model-path',
                        required=True,
                        help='filepath of the detection model to use.')

    parser.add_argument('-c', '--classification-model-path',
                        required=True,
                        help='filepath of the classification model to use.')

    args = parser.parse_args()

    class_names = ("Empty", "Full", "Garbage Bag")

    detector, _ = _build_detection_model(args.detection_model_path)
    classifier = _build_classification_model(args.classification_model_path)

    for input_file in args.input:
        im = Image.open(input_file)

        # path info for naming output file
        path = pathlib.Path(input_file)
        filename = path.stem
        extension = path.suffix

        cutouts = extract_predictions_from_image(detector, im, bounding_box_color=(255, 0, 0) if args.bounding_box else None)

        for idx, c in enumerate(cutouts):
            class_name, confidence = _classify(
                classifier, _image_to_normalized_tensor(c), class_names, minimal_confidence=0.5
            )

            outfile = args.output_mask.format(filename=filename, extension=extension[1:], index=idx, class_name=class_name)

            c.save(outfile)