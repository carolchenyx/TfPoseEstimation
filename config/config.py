# Copyright (c) Open-MMLab. All rights reserved.
import ast
import os.path as osp
import platform
import shutil
import sys
import tempfile
from argparse import Action, ArgumentParser
from addict import Dict
from importlib import import_module
import warnings


if platform.system() == 'Windows':
    import regex as re
else:
    import re

BASE_KEY = '_base_'
DELETE_KEY = '_delete_'
RESERVED_KEYS = ['filename', 'text', 'pretty_text']

class ConfigDict(Dict):

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no "
                                f"attribute '{name}'")
        except Exception as e:
            ex = e
        else:
            return value
        raise ex

class Config:
    # """A facility for config and config files.
    #
    # It supports common file formats as configs: python/json/yaml. The interface
    # is the same as a dict object and also allows access config values as
    # attributes.
    #
    # Example:
    #     >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
    #     >>> cfg.a
    #     1
    #     >>> cfg.b
    #     {'b1': [0, 1]}
    #     >>> cfg.b.b1
    #     [0, 1]
    #     >>> cfg = Config.fromfile('tests/data/config/a.py')
    #     >>> cfg.filename
    #     "/home/kchen/projects/mmcv/tests/data/config/a.py"
    #     >>> cfg.item4
    #     'test'
    #     >>> cfg
    #     "Config [path: /home/kchen/projects/mmcv/tests/data/config/a.py]: "
    #     "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"
    # """
    def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but '
                            f'got {type(cfg_dict)}')
        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f'{key} is reserved for config file')

        super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))
        super(Config, self).__setattr__('_filename', filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, 'r') as f:
                text = f.read()
        else:
            text = ''
        super(Config, self).__setattr__('_text', text)

    @staticmethod
    def _substitute_predefined_vars(filename, temp_config_name):
        file_dirname = osp.dirname(filename)
        file_basename = osp.basename(filename)
        file_basename_no_extension = osp.splitext(file_basename)[0]
        file_extname = osp.splitext(filename)[1]
        support_templates = dict(
            fileDirname=file_dirname,
            fileBasename=file_basename,
            fileBasenameNoExtension=file_basename_no_extension,
            fileExtname=file_extname)
        with open(filename, 'r') as f:
            config_file = f.read()
        for key, value in support_templates.items():
            regexp = r'\{\{\s*' + str(key) + r'\s*\}\}'
            value = value.replace('\\', '/')
            config_file = re.sub(regexp, value, config_file)
        with open(temp_config_name, 'w') as tmp_config_file:
            tmp_config_file.write(config_file)

    @staticmethod
    def fromfile(filename,
                 use_predefined_variables=True,
                 import_custom_modules=True):
        cfg_dict, cfg_text = Config._file2dict(filename,
                                               use_predefined_variables)
        if import_custom_modules and cfg_dict.get('custom_imports', None):
            Tool.import_modules_from_strings(**cfg_dict['custom_imports'])
        return Config(cfg_dict, cfg_text=cfg_text, filename=filename)

    @staticmethod
    def _validate_py_syntax(filename):
        with open(filename, 'r') as f:
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError('There are syntax errors in config '
                              f'file {filename}: {e}')

    @staticmethod
    def _file2dict(filename, use_predefined_variables=True):
        filename = osp.abspath(osp.expanduser(filename))
        Tool.check_file_exist(filename)
        fileExtname = osp.splitext(filename)[1]
        if fileExtname not in ['.py', '.json', '.yaml', '.yml']:
            raise IOError('Only py/yml/yaml/json type are supported now!')

        with tempfile.TemporaryDirectory() as temp_config_dir:
            temp_config_file = tempfile.NamedTemporaryFile(
                dir=temp_config_dir, suffix=fileExtname)
            if platform.system() == 'Windows':
                temp_config_file.close()
            temp_config_name = osp.basename(temp_config_file.name)
            # Substitute predefined variables
            if use_predefined_variables:
                Config._substitute_predefined_vars(filename,
                                                   temp_config_file.name)
            else:
                shutil.copyfile(filename, temp_config_file.name)

            if filename.endswith('.py'):
                temp_module_name = osp.splitext(temp_config_name)[0]
                sys.path.insert(0, temp_config_dir)
                Config._validate_py_syntax(filename)
                mod = import_module(temp_module_name)
                sys.path.pop(0)
                cfg_dict = {
                    name: value
                    for name, value in mod.__dict__.items()
                    if not name.startswith('__')
                }
                # delete imported module
                del sys.modules[temp_module_name]
            elif filename.endswith(('.yml', '.yaml', '.json')):
                import mmcv
                cfg_dict = mmcv.load(temp_config_file.name)
            # close temp file
            temp_config_file.close()

        cfg_text = filename + '\n'
        with open(filename, 'r') as f:
            cfg_text += f.read()

        if BASE_KEY in cfg_dict:
            cfg_dir = osp.dirname(filename)
            base_filename = cfg_dict.pop(BASE_KEY)
            base_filename = base_filename if isinstance(
                base_filename, list) else [base_filename]

            cfg_dict_list = list()
            cfg_text_list = list()
            for f in base_filename:
                _cfg_dict, _cfg_text = Config._file2dict(osp.join(cfg_dir, f))
                cfg_dict_list.append(_cfg_dict)
                cfg_text_list.append(_cfg_text)

            base_cfg_dict = dict()
            for c in cfg_dict_list:
                if len(base_cfg_dict.keys() & c.keys()) > 0:
                    raise KeyError('Duplicate key is not allowed among bases')
                base_cfg_dict.update(c)

            base_cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
            cfg_dict = base_cfg_dict

            # merge cfg_text
            cfg_text_list.append(cfg_text)
            cfg_text = '\n'.join(cfg_text_list)

        return cfg_dict, cfg_text




class Tool(Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options should
    be passed as comma separated values, i.e KEY=V1,V2,V3
    """
    def import_modules_from_strings(self,imports, allow_failed_imports=False):
        """Import modules from the given list of strings.

        # Args:
        #     imports (list | str | None): The given module names to be imported.
        #     allow_failed_imports (bool): If True, the failed imports will return
        #         None. Otherwise, an ImportError is raise. Default: False.
        #
        # Returns:
        #     list[module] | module | None: The imported modules.
        #
        # Examples:
        #     >>> osp, sys = import_modules_from_strings(
        #     ...     ['os.path', 'sys'])
        #     >>> import os.path as osp_
        #     >>> import sys as sys_
        #     >>> assert osp == osp_
        #     >>> assert sys == sys_
        # """
        if not imports:
            return
        single_import = False
        if isinstance(imports, str):
            single_import = True
            imports = [imports]
        if not isinstance(imports, list):
            raise TypeError(
                f'custom_imports must be a list but got type {type(imports)}')
        imported = []
        for imp in imports:
            if not isinstance(imp, str):
                raise TypeError(
                    f'{imp} is of type {type(imp)} and cannot be imported.')
            try:
                imported_tmp = import_module(imp)
            except ImportError:
                if allow_failed_imports:
                    warnings.warn(f'{imp} failed to import and is ignored.',
                                  UserWarning)
                    imported_tmp = None
                else:
                    raise ImportError
            imported.append(imported_tmp)
        if single_import:
            imported = imported[0]
        return imported

    def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
        if not osp.isfile(filename):
            raise FileNotFoundError(msg_tmpl.format(filename))




