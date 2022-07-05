### ©2020. Triad National Security, LLC. All rights reserved.
### This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
###
###

import os
import json

from dirtorch.utils.convenient import mkdir
from dirtorch.utils import common
import dirtorch.nets as nets
from dirtorch.datasets.generic import ImageListLabelsQ, ImageListLabels, ImageList
from dirtorch import test_dir

def load_model(path, iscuda, args):
    checkpoint = common.load_checkpoint(path, iscuda)

    # Set model options
    model_options={}
    model_options["arch"]=args.arch
    model_options["without_fc"] = True
    
    net = nets.create_model(pretrained="", **model_options)
    net = common.switch_model_to_cuda(net, iscuda, checkpoint)
    net.load_state_dict(checkpoint['state_dict'])

    net.preprocess = dict(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        input_size=args.image_size
    )
    
    if 'pca' in checkpoint:
        net.pca = checkpoint.get('pca')
    return net


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate a model')

    parser.add_argument('--dataset', '-d', type=str, required=True, help='Command to load dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to weights')

    # Model options
    parser.add_argument('--arch', type=str, required=True, help='Model architecture for testing')
    parser.add_argument('--image_size', type=int, default=224, help='default is 224')

    parser.add_argument('--trfs', type=str, required=False, default='', nargs='+', help='test transforms (can be several)')
    parser.add_argument('--pooling', type=str, default="gem", help='pooling scheme if several trf chains')
    parser.add_argument('--gemp', type=int, default=3, help='GeM pooling power')

    parser.add_argument('--out-json', type=str, default="", help='path to output json')
    parser.add_argument('--detailed', action='store_true', help='return detailed evaluation')
    parser.add_argument('--save-feats', type=str, default="", help='path to output features')
    parser.add_argument('--load-feats', type=str, default="", help='path to load features from')

    parser.add_argument('--threads', type=int, default=8, help='number of thread workers')
    #parser.add_argument('--cpu', default=False, action="store_true", help='run on CPU')
    parser.add_argument('--gpu', type=int, default=0, nargs='+', help='GPU ids')
    #parser.add_argument('--patents', default=False, action="store_true", help='run on patent data')
    #parser.add_argument('--sketches', default=False, action="store_true", help='run on sketch data')
    #parser.add_argument('--force-same', default=False, action="store_true", help='run on CPU')
    parser.add_argument('--dbg', default=(), nargs='*', help='debugging options')
    
    # post-processing
    parser.add_argument('--whiten', type=str, default=None, help='applies whitening')

    parser.add_argument('--aqe', type=int, nargs='+', help='alpha-query expansion paramenters')
    parser.add_argument('--adba', type=int, nargs='+', help='alpha-database augmentation paramenters')

    parser.add_argument('--whitenp', type=float, default=0.25, help='whitening power, default is 0.25 (i.e., 4th root)')
    parser.add_argument('--whitenv', type=int, default=None, help='number of components, default is None (i.e. all components)')
    parser.add_argument('--whitenm', type=float, default=1.0, help='whitening multiplier, default is 1.0 (i.e. no multiplication)')

    args = parser.parse_args()

    """ Define the dataset class
    """
    import pathlib
    fld = str( pathlib.Path(__file__).parent.parent.resolve() )
    if fld.startswith("/vast"):
        fld = fld.replace("/vast", "")

    class DeepPatentTest(ImageListLabelsQ):
        def __init__(self):
            ImageListLabelsQ.__init__(self, 
                    img_list_path=os.path.join(fld, "data/test_db_patent.txt"),
                    query_list_path=os.path.join(fld, "data/test_query_patent.txt"),
                    root=args.dataset )

    args.iscuda = common.torch_set_gpu(args.gpu)
    if args.aqe is not None:
        args.aqe = {'k': args.aqe[0], 'alpha': args.aqe[1]}
    if args.adba is not None:
        args.adba = {'k': args.adba[0], 'alpha': args.adba[1]}

    dataset = DeepPatentTest()
    print("Test dataset:", dataset)

    net = load_model(args.checkpoint, args.iscuda, args)

    if args.whiten:
        net.pca = net.pca[args.whiten]
        args.whiten = {'whitenp': args.whitenp, 'whitenv': args.whitenv, 'whitenm': args.whitenm}
    else:
        net.pca = None
        args.whiten = None

    # Evaluate
    res = test_dir.eval_model(dataset, net, args.trfs, pooling=args.pooling, gemp=args.gemp, detailed=args.detailed,
                     threads=args.threads, dbg=args.dbg, whiten=args.whiten, aqe=args.aqe, adba=args.adba,
                     save_feats=args.save_feats, load_feats=args.load_feats)
    
    if not args.detailed:
        print(' * ' + '\n * '.join(['%s = %g' % p for p in res.items()]))

    if args.out_json:
        # write to file
        try:
            data = json.load(open(args.out_json))
        except IOError:
            data = {}
        data[args.dataset] = res
        mkdir(args.out_json)
        open(args.out_json, 'w').write(json.dumps(data, indent=1))
        print("saved to "+args.out_json)
