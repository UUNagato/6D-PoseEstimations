'''
    This script is used to read in real data and output Error measure result.
'''

import numpy as np
import configparser
import argparse

import sys
sys.path.append('D:/MSRA/repos/Evaluation')
from sixd_toolkit.tools import eval_calc_errors, eval_loc

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--eval_dir', help='The folder of evaluated preditions', required=True)    # evaluation file folder
    # parser.add_argument('--model_path', default=None, required=True)    # the path to model
    parser.add_argument('--eval_cfg', help='The path of the evaluation config file', required=True)    # evaluation config file folder
    parser.add_argument('--scene_list', help='a python list contains the scene id you want to deal with', default=None)
    # not necessary, specify a specific scene to make program deal with a subset of scenes to avoid too high memory occupation.
    parser.add_argument('-s', '--silent', help='if silent, it will not output process image', default=True) # silent mode will not output any pictures.            
    parser.add_argument('-i', '--img_range', help='A tuple for image id range (left inclusive, right exclusive)', default=None)
    parser.add_argument('-se', '--summarize_error', help='whether or not to compute general error (use this after you compute all errors)', default=None)
    parser.add_argument('--only_summarize', help='This evaluation will only run final error summary.', default=None)

    arguments = parser.parse_args()

    eval_dir = arguments.eval_dir
    eval_args = configparser.ConfigParser(inline_comment_prefixes='#')
    eval_args.read(arguments.eval_cfg)      # read config file

    summary = False
    if arguments.summarize_error is not None:
        summary = eval(arguments.summarize_error)
    only_summarize = False
    if arguments.only_summarize is not None:
        print (arguments.only_summarize)
        only_summarize = eval(arguments.only_summarize)

    if only_summarize:
        if not eval_args.getboolean('EVALUATION','EVALUATE_ERRORS'):
            print ("Although only_summarize is specified, but the config file doesn't set EVALUATE_ERRORS to be true.")
            quit()
        eval_loc.match_and_eval_performance_scores(eval_args, eval_dir)
        quit()

    silent, scene_list, img_range = None, None, None
    if (arguments.silent != None):
        silent = eval(arguments.silent)
    if (arguments.scene_list != None):
        scene_list = eval(arguments.scene_list)
    if (arguments.img_range != None):
        img_range = eval(arguments.img_range)

    if (img_range != None and len(img_range) != 2):
        print ('the image range must be a tuple with two elements')
        quit()

    if (img_range != None and img_range[0] > img_range[1]):
        img_range = (img_range[1], img_range[0])

    print ("Some important parameters:\n eval_dir:{}".format(eval_dir))

    # now evaluate error
    if eval_args.getboolean('EVALUATION','COMPUTE_ERRORS'):
        print ("Start to call eval_calc_errors")
        if (scene_list != None):
            print ("Subset {} is evaluated".format(scene_list))
        eval_calc_errors.eval_calc_errors(eval_args, eval_dir, scene_list, img_range, silent)

    if summary and eval_args.getboolean('EVALUATION','EVALUATE_ERRORS'):
        print ("Start to call eval_loc.match_and_eval_performance_scores")
        eval_loc.match_and_eval_performance_scores(eval_args, eval_dir)

if __name__ == '__main__':
    main()