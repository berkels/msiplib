# pylint: disable=unsupported-membership-test,no-member

''' Module containing all necessary functions to process the configuration '''

import logging
import os
import sys
from datetime import datetime

import configargparse
import git

from msiplib.io import check_path

def parse_args():
    ''' function to parse arguments '''
    # If an arg is specified in more than one place, then commandline values override environment variables
    # which override config file values which override defaults.
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.ConfigparserConfigFileParser,
                                      description=\
                                      "Multiphase Mumford-Shah with Zach's or strongly convex convexification.")

    parser.add_argument('-c', '-configfile', dest='config_path', required=False, is_config_file=True,
                        help='Configuration file')
    parser.add_argument('--filename', dest='filename', required=True, help='Input image file')
    parser.add_argument('--gt', '--groundtruth', dest='gt_file', type=str, default=None,
                        help='Path to ground truth.')
    parser.add_argument('--numsegments', default=2, dest='k', type=int,
                        help='Number of segments.')
    parser.add_argument('--lambda', default=0.1, dest='reg_par', type=float,
                        help='Weight of the regularizer.')
    parser.add_argument('--irreg', dest='irreg', action='store_true',
                        help='If set, introduce an additional segment to collect irregular points.')
    parser.add_argument('--radius', dest='r', default=1, type=int,
                        help='Radius of balls used to find irregular segment.')
    parser.add_argument('--irregeps', dest='irreg_eps', default=1e-06, type=float,
                        help='Epsilon needed to compute indicator value of irregular region (cf. thesis of Mevenkamp).')
    parser.add_argument('--indicatorfunction', '--indfunc', dest='ind_func', default='2', type=str,
                        choices=['2', 'anisotropic-eps-inverse', 'anisotropic-eps-normal', 'anisotropic-eps-discard',
                                 'non-squared-anisotropic-eps-inverse', 'kernel', 'kernelpca-eps-inverse',
                                 'kernelpca-eps-normal'],
                        help='Indicator function.')
    parser.add_argument('--indeps', default=0.1, dest='ind_eps', type=float, nargs='?',
                        help='Epsilon to regularize indicator function.')
    parser.add_argument('--kernel', dest='kernel', default='gaussian', type=str,
                        choices=['gaussian', 'polynomial', 'chi-squared', 'laplacian', 'cosine', 'sigmoid'],
                        help='Kernel function.')
    parser.add_argument('--indparams', default='0.01', nargs='+', dest='ind_params', type=float,
                        help='List of parameters for indicator function.')
    parser.add_argument('--convexification', dest='convexification', default='zach', type=str,
                        choices=['zach', 'strongly-conv', 'strongly-conv-uc'],
                        help='Convexification of Mumford-Shah segmentation functional.')
    parser.add_argument('--pdhgtype', dest='pdhgtype', default=1, type=int, choices=[1, 2],
                        help='Type of primal dual hybrid gradient algorithm used for minimization.')
    parser.add_argument('--precision', dest='precision', default='float', type=str, choices=['float', 'double'],
                        help='Computing precision.')
    parser.add_argument('--maxiter', default=1000, dest='max_iter', type=int,
                        help='Maximal number of iterations.')
    parser.add_argument('--maxouteriter', default=20, dest='outer_iter', type=int,
                        help='Maximal number of outer iterations.')
    parser.add_argument('--stopeps', default=1e-06, dest='stop_eps', type=float,
                        help='Threshold of the inner stopping criterion.')
    parser.add_argument('--outerstopeps', default=1e-06, dest='outer_stop_eps', type=float,
                        help='Threshold of the outer stopping criterion.')
    parser.add_argument('--bandselection', dest='band_selection', default='full', type=str,
                        choices=['full', 'issc'], help='Band selection method')
    parser.add_argument('--bandselectparam', dest='band_select_param', default=0.0001, type=float,
                        help='Parameter of band selection method')
    parser.add_argument('--reducedim', dest='reduce_dimensionality', action='store_true',
                        help='If set, reduce dimensionality of data before processing it.')
    parser.add_argument('--dimredmethod', dest='dim_red_method', default='pca', type=str,
                        choices=['pca', 'tsne', 'umap', 'isomap', 'tga', 'mnf', 'ica'],
                        help='Algorithm to reduce dimensionality before minimizing MS functional.')
    parser.add_argument('--keepnumfeatures', dest='n_features', default=0.999, type=float,
                        help='Number of features to keep with dimensionality reduction method.')
    parser.add_argument('--init', dest='init', default='kmeans', type=str,
                        choices=['kmeans', 'random', 'dbscan', 'gmm', 'hierarchical', 'optics', 'birch', 'gt'],
                        help='Choose method to find initial segmentation.')
    parser.add_argument('--reducediminit', dest='reduce_dim_init', action='store_true',
                        help='If set, run initialization on data after reducing dimensionality.')
    parser.add_argument('--means', dest='means', default='arithmetic', type=str,
                        choices=['arithmetic', 'trimmed'],
                        help='Algorithm that provides mean values for indicator function.')
    parser.add_argument('--variances', dest='variances', default='sdm', type=str, choices=['sdm', 'trimmed'],
                        help='Method that provides variances for anisotropic 2-norm.')
    parser.add_argument('--components', dest='components', default='pca', type=str, choices=['pca', 'tga', 'mnf'],
                        help='Algorithm that provides directions for anisotropic 2-norm.')
    parser.add_argument('--trimproportion', dest='trim_proportion', default=0.5, type=float,
                        help='Proportion of points that are trimmed at both sides before computing.')
    parser.add_argument('--seed', dest='seed', type=int, default=42,
                        help='Seed for randomness.')
    parser.add_argument('--ignorelabel', dest='ignore_label', type=int, default=None,
                        help='If ground truth provided, ignore pixels carrying this label.')
    parser.add_argument('--ignorepixels', dest='ignore_pixels', action='store_true',
                        help='If set, ignore pixels with ignore label also in computations.')
    parser.add_argument('--ignorepixelsdataterm', dest='ignore_pixels_data_term', action='store_true',
                        help='If set, ignore pixels with ignore label in data term.')
    parser.add_argument('--solveemptysegment', dest='solve_empty_segment', action='store_true',
                        help='If set, try solve problem with empty segments instead of stopping iterations.')
    parser.add_argument('--saveintermediatesegs', dest='save_intermediate_segs', action='store_true',
                        help='If set, save intermediate segmentations in each iteration.')

    args = parser.parse_args()

    args.filename = os.path.expandvars(args.filename)
    if args.gt_file is not None:
        args.gt_file = os.path.expandvars(args.gt_file)

    return vars(args)

def initialize_logger(args):
    ''' initialization of logger'''
    path = os.path.expandvars('${OUTPUT_DIR}/segmentation/{}') \
			.format(map_ind_func_to_folder_name(args['ind_func'], args['kernel']))
    image_index = get_image_index(args['filename'])
    path = f'{path}/{image_index}'

    if not os.path.exists(path):
        check_path(path)
        print(f'Created directory {path}.')
    now = datetime.now()
    time_stamp = f'{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-'\
                 f'{now.second}'
    folder_name = f"{image_index}-{time_stamp}_"\
                  f"lam{str(args['reg_par']).replace('.', '_')}_"\
                  f"indp{str(args['ind_eps']).replace('.', '_').replace(' ', '_').replace(',', '_').replace('[','').replace(']','')}_"\
                  f"nfeat{str(int(args['n_features']))}"
    try:
        os.mkdir(f'{path}/{folder_name}')
    except OSError:
        print(f'Creation of directory {folder_name} failed.')
        print(f'Write files to {path} instead.')
    else:
        print(f'Creation of directory {folder_name} was successful.')
        path = f'{path}/{folder_name}'

    filepath_name = f"{path}/lam{str(args['reg_par']).replace('.', '_')}_" \
                    f"indp{str(args['ind_eps']).replace('.', '_').replace(' ', '_').replace(',', '_').replace('[','').replace(']','')}" \
                    f"_nfeat{str(int(args['n_features']))}"
    filename = f'{filepath_name}.log'

    git_info = get_git_info()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=filename
                       )
    ms_logger = logging.getLogger('MS')
    ms_logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    ms_logger.addHandler(console_handler)

    ms_logger.info('Perform an image segmentation using the Mumford-Shah model with %s convexification.',
                   args['convexification'])
    ms_logger.info('Time stamp: %s', time_stamp)
    ms_logger.info('Working on: %s', os.uname()[1])
    ms_logger.info('Current msiplib git branch: %s', git_info['msiplib']['branch'])
    ms_logger.info('Current msiplib git commit: %s', git_info['msiplib']['commit'])
    ms_logger.info('Configuration:')
    ms_logger.info('Number of segments: %s', args['k'])
    ms_logger.info('Regularization parameter: %s', args['reg_par'])
    ms_logger.info('Precision: %s', args['precision'])
    ms_logger.info('Maximum number of PDHG iterations: %s', args['max_iter'])
    ms_logger.info('Maximum number of outer iterations: %s', args['outer_iter'])
    ms_logger.info('Stopping threshold for PDHG: %s', args['stop_eps'])
    ms_logger.info('Stopping threshold for outer iterations: %s', args['outer_stop_eps'])
    ms_logger.info('Convexification: %s', args['convexification'])
    ms_logger.info('PDHG type: %s', args['pdhgtype'])
    ms_logger.info('Stopping criterion: weighted mean of inf-norm of mean feature values difference')

    ms_logger.info('Computation of means: %s', args['means'])
    if args['means'] == 'trimmed':
        ms_logger.info('Trim proportion for means: %s', args['trim_proportion'])

    if args['ind_func'] == '2':
        # the typical 2-norm is used as indicator function
        ms_logger.info('Indicator function: 2-norm.')
    elif args['ind_func'] == 'anisotropic-eps-normal':
        # the anisotropic 2-norm with regularization with epsilon is used as indicator function
        ms_logger.info('Indicator function: anisotropic-eps-normal')
        ms_logger.info('Epsilon: %s', args['ind_eps'])
        ms_logger.info('Computation of components: %s', args['components'])
        if args['components'] == 'tga':
            ms_logger.info('Trim proportion for TGA: %s', args['trim_proportion'])
        elif args['components'] == 'mnf':
            ms_logger.info('Noise estimation for MNF: shift differences within bands')
            ms_logger.info('Shift direction: lower')
        ms_logger.info('Computation of variances: %s', args['variances'])
        if args['variances'] == 'trimmed':
            ms_logger.info('Trim proportion for variances: %s', args['trim_proportion'])
    elif args['ind_func'] == 'anisotropic-eps-discard':
        # the anisotropic 2-norm discarding directions with small standard deviation is used as indicator function
        ms_logger.info('Indicator function: anisotropic-eps-discard')
        ms_logger.info('Epsilon: %s', args['ind_eps'])
        ms_logger.info('Computation of components: %s', args['components'])
        ms_logger.info('Computation of variances: %s', args['variances'])
    elif args['ind_func'] == 'anisotropic-eps-inverse':
        # the anisotropic 2-norm with regularization with 1/epsilon is used as indicator function
        ms_logger.info('Indicator function: anisotropic-eps-inverse')
        ms_logger.info('Epsilon: %s', args['ind_eps'])
        ms_logger.info('Computation of components: %s', args['components'])
        if args['components'] == 'tga':
            ms_logger.info('Trim proportion for TGA: %s', args['trim_proportion'])
        elif args['components'] == 'mnf':
            ms_logger.info('Noise estimation for MNF: shift differences within bands')
            ms_logger.info('Shift direction: lower')
        ms_logger.info('Computation of variances: %s', args['variances'])
        if args['variances'] == 'trimmed':
            ms_logger.info('Trim proportion for variances: %s', args['trim_proportion'])
    elif args['ind_func'] == 'non-squared-anisotropic-eps-inverse':
        # the non-squared anisotropic 2-norm with regularization with 1/epsilon is used as indicator function
        ms_logger.info('Indicator function: non-squared anisotropic-eps-inverse')
        ms_logger.info('Epsilon: %s', args['ind_eps'])
        ms_logger.info('Maximum number of fixed point iterations: %s', args['ind_params'][0])

    if args['ignore_label'] is not None:
        ms_logger.info('Ignore label: %s', args['ignore_label'])
        ms_logger.info('Ignore pixels in computations: %s', args['ignore_pixels'])
        ms_logger.info('Ignore pixels in data term: %s', args['ignore_pixels_data_term'])
    ms_logger.info('Band selection method: %s', args['band_selection'])
    if args['band_selection'] != 'full':
        ms_logger.info('Parameter of band selection method: %s', args['band_select_param'])
    ms_logger.info('Reduce dimensionality: %s', args['reduce_dimensionality'])
    if args['reduce_dimensionality']:
        ms_logger.info('Dimensionality reduction method: %s', args['dim_red_method'])
        ms_logger.info('Number of features to keep: %s', args['n_features'])
    ms_logger.info('Reduce dimensionality before initialization: %s', args['reduce_dim_init'])
    ms_logger.info('Initialization method: %s', args['init'])
    ms_logger.info('Seed for randomness: %s', args['seed'])
    ms_logger.info('Solve empty segment: %s', args['solve_empty_segment'])

    return ms_logger, filepath_name

def map_ind_func_to_folder_name(ind_func, kernel=None):
    ''' maps chosen indicator function to corresponding folder name for logging '''
    if ind_func == '2' or ind_func == '2-norm':
        folder = '2norm'
    elif ind_func == 'anisotropic-eps-inverse':
        folder = 'anisotropic_2norm_eps_inverse'
    elif ind_func == 'anisotropic-eps-normal':
        folder = 'anisotropic_2norm_eps_normal'
    elif ind_func == 'anisotropic-eps-discard':
        folder = 'anisotropic_2norm_eps_discard'
    elif ind_func == 'non-squared-anisotropic-eps-inverse':
        folder = 'non-squared_anisotropic_2norm_eps_inverse'

    return folder


def get_image_index(filename):
    ''' functions extracts image index from filename '''
    name = filename.split('/')[-1]
    # Botswana
    if name.find('Botswana') >= 0:
        index = 'Botswana'

    # ftir_zero
    elif name.find('ftir_zero') >= 0:
        index = 'ftir_zero'

    # ftir
    elif name.find('ftir') >= 0:
        index = 'ftir'

    # IndianPines
    elif name.find('IndianPines') >= 0:
        index = 'IndianPines'

    # KennedySpaceCenter
    elif name.find('KennedySpaceCenter') >= 0:
        index = 'KennedySpaceCenter'

    # PaviaCenter
    elif name.find('PaviaCenter') >= 0:
        index = 'PaviaCenter'

    # PaviaUniversity
    elif name.find('PaviaUniversity') >= 0:
        index = 'PaviaUniversity'

    # Salinas, SalinasA, SalinasA_mod
    elif name.find('Salinas') >= 0:
        if name.find('SalinasA') >= 0:
            if name.find('_mod') >= 0:
                index = 'SalinasA_mod'
            else:
                index = 'SalinasA'
        else:
            index = 'Salinas'

    # Synthetic images
    elif name.find('synthetic') >= 0:
        start = name.find('image') + 5
        end = (name.find('_', start)
               if (name.find('_', start) >= 0) and (name.find('_', start) < name.find('.', start))
               else name.find('.'))
        end = name.find('.') if name.find('_', start) < name.find('.', start) else name.find('_', start)
        index = name[start : end]

    # Urban
    elif name.find('Urban') >= 0:
        index = 'Urban'

    # all other cases
    else:
        index = 0

    if name.find('reduced') >= 0:
        index = f'{index}_rpca'

    if '3dcae-fl' in name:
        index = f'{index}_3dcae-fl'
    return index


def create_config(filename, args):
    '''
        creates and stores a configuration file to run MS segmentation based on given arguments
        Args:
            filename: filename where configuration file should be stored
            args: dictionary containing the values of the arguments for the MS segmentation framework
    '''

    # parameter name correction
    translation_dict = {
        'k': 'numsegments',
        'gt_file': 'groundtruth',
        'reg_par': 'lambda',
        'r': 'radius',
        'irreg_eps': 'irregeps',
        'ind_func': 'indicatorfunction',
        'ind_eps': 'indeps',
        'ind_params': 'indparams',
        'max_iter': 'maxiter',
        'outer_iter': 'maxouteriter',
        'stop_eps': 'stopeps',
        'outer_stop_eps': 'outerstopeps',
        'band_selection': 'bandselection',
        'band_select_param': 'bandselectparam',
        'reduce_dimensionality': 'reducedim',
        'dim_red_method': 'dimredmethod',
        'n_features': 'keepnumfeatures',
        'reduce_dim_init': 'reducediminit',
        'trim_proportion': 'trimproportion',
        'ignore_label': 'ignorelabel',
        'ignore_pixels': 'ignorepixels',
        'ignore_pixels_data_term': 'ignorepixelsdataterm',
        'solve_empty_segment': 'solveemptysegment',
        'save_intermediate_segs': 'saveintermediatesegs'
    }

    # add elements from args dictionary to translated version
    external_args = {}
    for key, value in args.items():
        # if internal parameter differs from external one, add parameter value with external key
        if key in translation_dict:
            external_args[translation_dict[key]] = value
        else:
            external_args[key] = value

    # remove config_path from external_args since it is not needed
    external_args.pop('config_path')

    # remove ignorelabel from external_args if it was not given and is hence None
    if external_args['ignorelabel'] is None:
        external_args.pop('ignorelabel')

    # creating a config file and storing at filename
    data_dir = os.path.expandvars('${DATA_DIR}')
    with open(filename, 'w', encoding='utf_8') as f:
        f.write('[parameters]\n')
        # for each parameter in arguments write it and its value to the file
        for key, value in external_args.items():
            # when processing the input filename or the ground truth file,
            # change absolute path to environment variable $DATA_DIR
            if key in ('filename', 'groundtruth'):
                f.write(f'{key} = {value.replace(data_dir, "${DATA_DIR}")}\n')
            else:
                f.write(f'{key} = {value}\n')

        f.close()

    # write git info to parameter file
    append_git_info(filename)


def get_git_info():
    ''' function extracts information about current git branch and hash of current commit '''
    git_info = {}

    # msiplib repository
    msiplib_repo = git.Repo(path=os.path.expandvars('${REPOS_DIR}/msip/msiplib'))
    git_info['msiplib'] = {'branch': msiplib_repo.active_branch.name, 'commit': msiplib_repo.head.object.hexsha}

    return git_info


def append_git_info(file):
    ''' function appends git info to a given file '''
    git_info = get_git_info()
    with open(file, 'a', encoding='utf_8') as f:
        f.write(f'# Current msiplib git branch: {git_info["msiplib"]["branch"]}\n')
        f.write(f'# Current msiplib git commit: {git_info["msiplib"]["commit"]}\n')
        f.close()
