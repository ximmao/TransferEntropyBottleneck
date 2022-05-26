import yaml
import os
import copy
from dataset import *
from models import VAEModel, VAEModel_BE, CVAEModel, TEModel

class structArgs():
    def __init__(self, a_args):
        for arg_name in vars(a_args):
            setattr(self, arg_name, getattr(a_args, arg_name))

    def update_as_dict(self, other_dict):
        for k, v in other_dict.items():
            setattr(self, k, v)
                
def getStructuredArgs(yaml_file, argparse_args):
    if not os.path.exists(yaml_file):
        print('file not found')
        exit(1)
    stream = open(yaml_file, 'r')
    load_dict = yaml.load(stream, yaml.Loader)
    
    # copy all argparse arguments
    custArgs = structArgs(argparse_args)
    
    # incorporate all yaml argments
    dset_yaml = load_dict['dataset_config']
    mod_yaml = load_dict['model_config']
    gen_yaml = load_dict['general_config']
    if dset_yaml['dataset_class'] == 'ColoredBouncingBallsStackedOnlinegen':
        custArgs.dataset_class = ColoredBouncingBallsStackedOnlinegen
    elif dset_yaml['dataset_class'] == 'VariedRotatingDigitsOnlinegen':
        custArgs.dataset_class = VariedRotatingDigitsOnlinegen
    else:
        raise ValueError()
    custArgs.dataset_name = dset_yaml['dataset_class']
    assert isinstance(custArgs.dataset_name, str)

    if mod_yaml['Y_module_type'] == 'VAEModel':
        custArgs.Y_module_type = VAEModel
    elif mod_yaml['Y_module_type'] == 'VAEModel_BE':
        custArgs.Y_module_type = VAEModel_BE
    else:
        raise ValueError()

    if mod_yaml['X_module_type'] == 'CVAEModel':
        custArgs.X_module_type = CVAEModel
    else:
        raise ValueError()
    
    custArgs.update_as_dict(gen_yaml)
    del custArgs.batch_size
    custArgs.train_batch_size = gen_yaml['batch_size']['train']
    custArgs.test_batch_size = gen_yaml['batch_size']['test']
    assert custArgs.loss_type in ['mse','binary','gaussian','l1','laplace'], 'valid loss type in '+str(['mse','binary','gaussian','l1','laplace'])
    
    # train / valid / test configs
    d_trainset_argu = copy.deepcopy(dset_yaml)
    if 'data_path' in d_trainset_argu:
        del d_trainset_argu['data_path']
        d_trainset_argu['directory'] = dset_yaml['data_path']['train']
    del d_trainset_argu['dataset_class']
    d_trainset_argu['split_name'] = 'train'
    custArgs.trainset_argu = d_trainset_argu

    d_validset_argu = copy.deepcopy(dset_yaml)
    if 'data_path' in d_validset_argu:
        del d_validset_argu['data_path']
        d_validset_argu['directory'] = dset_yaml['data_path']['valid']
    del d_validset_argu['dataset_class']
    d_validset_argu['split_name'] = 'valid'
    custArgs.validset_argu = d_validset_argu

    d_testset_argu = copy.deepcopy(dset_yaml)
    if 'data_path' in d_testset_argu:
        del d_testset_argu['data_path']
        d_testset_argu['directory'] = dset_yaml['data_path']['test']
    del d_testset_argu['dataset_class']
    d_testset_argu['split_name'] = 'test'
    custArgs.testset_argu = d_testset_argu

    # model configs
    d_y_module_argu = mod_yaml['Y_module_args']
    d_x_module_argu = mod_yaml['X_module_args']

    d_y_module_argu['latent_dim'] = mod_yaml['TE_model_exclu']['latent_dim']
    d_x_module_argu['latent_dim'] = mod_yaml['TE_model_exclu']['latent_dim']

    #these model params are properties of the dataset
    if dset_yaml['dataset_class'] == 'VariedRotatingDigitsOnlinegen':
        d_y_module_argu.update({'is_2d':True,'nc': 1, 'output_res' : 28, 'oc' : 1, 'encoder_type' : 'lstm_resnet18_2d'})
        d_x_module_argu.update({'is_2d':True,'input_dim': 10, 'output_res' : 28, 'oc' : 1,
         'encoder_type' : 'lstm_embed',})
    elif dset_yaml['dataset_class'] == 'ColoredBouncingBallsStackedOnlinegen':
        if dset_yaml['baseline_train']:
            new_nc_y = (int(dset_yaml['num_distractor']) + 1) * 3
        else:
            new_nc_y = int(dset_yaml['num_distractor']) * 3
        new_nc_x = (int(dset_yaml['num_distractor']) + 1) * 3
        d_y_module_argu.update({'is_2d':True,'nc': new_nc_y, 'output_res' : 32, 'oc' : 3, 'encoder_type' : 'lstm_resnet18_2d'})
        d_x_module_argu.update({'is_2d':True,'input_dim': new_nc_x, 'output_res' : 32, 'oc' : 3,
         'encoder_type' : 'lstm_resnet18_2d',})
    else:
        raise ValueError()

    #output_categorical
    mod_yaml['TE_model_exclu'].update({'output_categorical':gen_yaml['output_categorical']})
    d_x_module_argu.update({'output_categorical':gen_yaml['output_categorical']})
    d_y_module_argu.update({'output_categorical':gen_yaml['output_categorical']})
    if mod_yaml['TE_model_exclu']['output_categorical']:
        d_y_module_argu.update({'nc': d_x_module_argu['input_dim'],'oc': d_x_module_argu['input_dim']})

    #teb0_nocontext_mlp_conditional
    mod_yaml['TE_model_exclu'].update({'teb0_nocontext_mlp_conditionals':gen_yaml['teb0_nocontext_mlp_conditionals']})
    d_x_module_argu.update({'teb0_nocontext_mlp_conditionals':gen_yaml['teb0_nocontext_mlp_conditionals']})
    d_y_module_argu.update({'teb0_nocontext_mlp_conditionals':gen_yaml['teb0_nocontext_mlp_conditionals']})

    if d_y_module_argu['is_2d'] == True:
        res = d_y_module_argu['output_res'] 
        d_y_module_argu['output_dim'] = (res, res)
        del d_y_module_argu['output_res']
        custArgs.image_shape = (d_y_module_argu['oc'], res, res)
    else:
        custArgs.image_shape = (1, res)

    # optional:
    d_y_module_argu['dec_multiplier']=(2, 2)
    d_y_module_argu['dec_pad']=(0,0)
    custArgs.Y_module_args_dict = d_y_module_argu

    d_x_module_argu = mod_yaml['X_module_args']
    if d_x_module_argu['is_2d'] == True:
        res = d_x_module_argu['output_res'] 
        d_x_module_argu['output_dim'] = (res, res)
        del d_x_module_argu['output_res']
    # optional:
    d_x_module_argu['dec_multiplier']=(2, 2)
    d_x_module_argu['dec_pad']=(0,0)


    d_te_module_argu = {'Y_module_args_dict': d_y_module_argu,
                        'X_module_args_dict': d_x_module_argu}
    d_te_module_argu.update(mod_yaml['TE_model_exclu'])
    custArgs.TE_module_args_dict = d_te_module_argu

    del load_dict
    return custArgs