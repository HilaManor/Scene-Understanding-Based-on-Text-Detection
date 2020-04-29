# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
#
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Imports
import argparse
import os
import cv2

from panorama_maker import PanoramaMaker


def parse_dir(scene_path, output_path):
    """ Runs the context analysis on the given images of the scene in the given directory

    :param scene_path: Directory containing images of the scene. images should overlap
    :param output_path: All of the algorithm output will be thrown out in this path
    :return: None
    """
    panorama_gen = PanoramaMaker()
    for im_name in sorted(os.listdir(scene_path)):
        print("Working on \"%s\"..." % im_name)
        im_file = os.path.join(scene_path, im_name)
        im_original = cv2.imread(im_file)

        panorama_gen.add_photo(im_original)
        # TODO - hmm?
        # im, scale_w, scale_h, original_w, original_h = resize(im_original,
        #                                                       size=cfg.INPUT_SIZE)

    panorama = panorama_gen.create_panorama()
    # TODO - success rate or something for panorama


if __name__ == '__main__':
    print("Gathering Data...")
    # ----- Creating Argument Parser -----
    parser = argparse.ArgumentParser(description="Main Runner")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--scenes_dir", type=str,
                             help="path to multiple scenes parent directory")
    input_group.add_argument("--single_scene", type=str,
                             help="path to a directory containing one scene")
    parser.add_argument("--results_dir", type=str,
                        help="Results directory. "
                             "Default is to throw the output inside given input directory")

    text_net_group = parser.add_argument_group('Text Recognition Network',
                                               'arguments related to the text '
                                               'detection and recognition network')
    text_net_group.add_argument("--config file", help="path to config file", type=str,
                                default='.\\research-charnet-master\\configs'
                                        '\\icdar2015_hourglass88.yaml')

    args = parser.parse_args()
    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)

    if args.scenes_dir:
        if not args.results_dir:
            args.results_dir = args.scenes_dir
        for scene in sorted(os.listdir(args.scenes_dir)):
            print("\nProcessing \"%s\":" % scene)
            curr_scene_path = os.path.join(args.scenes_dir, scene)
            curr_output_dir = os.path.join(args.results_dir, scene)
            os.makedirs(curr_output_dir, exist_ok=True)
            parse_dir(curr_scene_path, curr_output_dir)
    else:
        if not args.results_dir:
            args.results_dir = args.single_scene
        parse_dir(args.single_scene, args.results_dir)
