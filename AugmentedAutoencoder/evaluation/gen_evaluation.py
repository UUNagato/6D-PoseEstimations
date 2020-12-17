'''
    This script is used to generate .bat or .sh file to evaluate prediction in batch.
'''

import argparse
import os

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-ib','--img_batch',help="How many images processed for each scene",type=int,default=600)
    parser.add_argument('-maxi','--max_imgs',help="How many images in total for each scene",type=int,default=600)
    parser.add_argument('-iblist', '--img_batchlist', help="Specify batch size for each scene",default=None)
    parser.add_argument('-sb','--scene_batch',help="How many scenes processed for each process",type=int,default=1)
    parser.add_argument('-maxs', '--max_scenes', help="How many scenes in total",type=int,default=20)
    parser.add_argument('-slist', '--scene_list', help="A list of scene ids, if specified, --max_scenes will be ignored", default=None)

    parser.add_argument('--eval_dir', help="The prediction files location (the folder containing scene numbered prediction folders)", required=True)
    parser.add_argument('--eval_cfg', help="The evaluation config file", required=True)

    parser.add_argument('-o', '--output', help="Output batch file",default='./evaluate.bat')
    parser.add_argument('-s', '--silent', help="If some evaluation images should be output", default=False, type=bool)
    arguments = parser.parse_args()

    output = arguments.output
    img_batch = arguments.img_batch
    max_imgs = arguments.max_imgs
    scene_batch = arguments.scene_batch
    max_scenes = arguments.max_scenes
    silent = arguments.silent

    slist = None
    iblist = None
    if arguments.scene_list is not None:
        slist = eval(arguments.scene_list)
    if arguments.img_batchlist is not None:
        iblist = eval(arguments.img_batchlist)

    eval_cfg = arguments.eval_cfg
    eval_dir = arguments.eval_dir

    if not os.path.exists(eval_dir):
        print ("Warning, the eval_dir input doesn't exist: %s" % eval_dir)

    if not os.path.exists(eval_cfg):
        print ("Warning, the eval_cfg input doesn't exist: %s" % eval_cfg)

    if img_batch <= 0 or max_imgs <= 0 or scene_batch <= 0 or max_scenes <= 0:
        print ("Please make sure img_batch, max_imgs, scene_batch and max_scenes are all positive. Their values are:%d, %d, %d, %d" % (img_batch, max_imgs, scene_batch, max_scenes))
        quit()
    
    if slist != None:
        print ("Scene list:{} is used".format(slist))
    else:
        print ("Max scene id:%d" % max_scenes)

    commands = 0
    with open(output, 'w') as f:
        base_command = 'python evaluation.py --eval_dir="{}" --eval_cfg="{}" --scene_list="{}" --img_range="{}" --silent={} --summarize_error={} || exit /b\n'
        # python command and terminate when an error happenes.\
        LoopList = range(1, max_scenes+1, scene_batch)
        if slist is not None:
            LoopList = slist

        for sid in LoopList:
            next_scene_start = min(max_scenes + 1, sid + scene_batch)
            scene_list = list(range(sid, next_scene_start))
            if iblist is not None:
                img_batch = iblist[sid - 1]

            for iid in range(0, max_imgs, img_batch):
                img_range = (iid, min(max_imgs + 1, iid + img_batch))
                if sid + scene_batch >= max_scenes + 1 and iid + img_batch >= max_imgs:
                    command = base_command.format(eval_dir, eval_cfg, scene_list, img_range, silent, True)
                else:
                    command = base_command.format(eval_dir, eval_cfg, scene_list, img_range, silent, False)
                f.write(command)
                commands += 1

                if commands % 20 == 0:
                    f.write("ping -n 5 127.0.0.1>nul\n")      # make it rest for a while to see if it helps solve crashing.

if __name__ == '__main__':
    main()