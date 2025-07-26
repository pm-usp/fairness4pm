# Contains the helper functions for the optim_preproc class
import numpy as np
import pandas as pd


def get_distortion_adult(vold, vnew):
    """Distortion function for the adult dataset. We set the distortion
    metric here. See section 4.3 in supplementary material of
    http://papers.nips.cc/paper/6988-optimized-pre-processing-for-discrimination-prevention
    for an example

    Note:
        Users can use this as templates to create other distortion functions.

    Args:
        vold (dict) : {attr:value} with old values
        vnew (dict) : dictionary of the form {attr:value} with new values

    Returns:
        d (value) : distortion value
    """

    # Define local functions to adjust education and age
    def adjustEdu(v):
        if v == '>12':
            return 13
        elif v == '<6':
            return 5
        else:
            return int(v)

    def adjustAge(a):
        if a == '>=70':
            return 70.0
        else:
            return float(a)

    def adjustInc(a):
        if a == "<=50K":
            return 0
        elif a == ">50K":
            return 1
        else:
            return int(a)

    # value that will be returned for events that should not occur
    bad_val = 3.0

    # Adjust education years
    eOld = adjustEdu(vold['Education Years'])
    eNew = adjustEdu(vnew['Education Years'])

    # Education cannot be lowered or increased in more than 1 year
    if (eNew < eOld) | (eNew > eOld+1):
        return bad_val

    # adjust age
    aOld = adjustAge(vold['Age (decade)'])
    aNew = adjustAge(vnew['Age (decade)'])

    # Age cannot be increased or decreased in more than a decade
    if np.abs(aOld-aNew) > 10.0:
        return bad_val

    # Penalty of 2 if age is decreased or increased
    if np.abs(aOld-aNew) > 0:
        return 2.0

    # Adjust income
    incOld = adjustInc(vold['Income Binary'])
    incNew = adjustInc(vnew['Income Binary'])

    # final penalty according to income
    if incOld > incNew:
        return 1.0
    else:
        return 0.0


def get_distortion_german(vold, vnew):
    """Distortion function for the german dataset. We set the distortion
    metric here. See section 4.3 in supplementary material of
    http://papers.nips.cc/paper/6988-optimized-pre-processing-for-discrimination-prevention
    for an example

    Note:
        Users can use this as templates to create other distortion functions.

    Args:
        vold (dict) : {attr:value} with old values
        vnew (dict) : dictionary of the form {attr:value} with new values

    Returns:
        d (value) : distortion value
    """

    # Distortion cost
    distort = {}
    distort['credit_history'] = pd.DataFrame(
                                {'None/Paid': [0., 1., 2.],
                                'Delay':      [1., 0., 1.],
                                'Other':      [2., 1., 0.]},
                                index=['None/Paid', 'Delay', 'Other'])
    distort['employment'] = pd.DataFrame(
                            {'Unemployed':    [0., 1., 2.],
                            '1-4 years':      [1., 0., 1.],
                            '4+ years':       [2., 1., 0.]},
                            index=['Unemployed', '1-4 years', '4+ years'])
    distort['savings'] = pd.DataFrame(
                            {'Unknown/None':  [0., 1., 2.],
                            '<500':           [1., 0., 1.],
                            '500+':           [2., 1., 0.]},
                            index=['Unknown/None', '<500', '500+'])
    distort['status'] = pd.DataFrame(
                            {'None':          [0., 1., 2.],
                            '<200':           [1., 0., 1.],
                            '200+':           [2., 1., 0.]},
                            index=['None', '<200', '200+'])
    distort['credit'] = pd.DataFrame(
                        {'Bad Credit' :    [0., 1.],
                         'Good Credit':    [2., 0.]},
                         index=['Bad Credit', 'Good Credit'])
    distort['sex'] = pd.DataFrame(
                        {0.0:    [0., 2.],
                         1.0:    [2., 0.]},
                         index=[0.0, 1.0])
    distort['age'] = pd.DataFrame(
                        {0.0:    [0., 2.],
                         1.0:    [2., 0.]},
                         index=[0.0, 1.0])

    total_cost = 0.0
    for k in vold:
        if k in vnew:
            total_cost += distort[k].loc[vnew[k], vold[k]]

    return total_cost

def get_distortion_compas(vold, vnew):
    """Distortion function for the compas dataset. We set the distortion
    metric here. See section 4.3 in supplementary material of
    http://papers.nips.cc/paper/6988-optimized-pre-processing-for-discrimination-prevention
    for an example

    Note:
        Users can use this as templates to create other distortion functions.

    Args:
        vold (dict) : {attr:value} with old values
        vnew (dict) : dictionary of the form {attr:value} with new values

    Returns:
        d (value) : distortion value
    """
    # Distortion cost
    distort = {}
    distort['two_year_recid'] = pd.DataFrame(
                                {'No recid.':     [0., 2.],
                                'Did recid.':     [2., 0.]},
                                index=['No recid.', 'Did recid.'])
    distort['age_cat'] = pd.DataFrame(
                            {'Less than 25':    [0., 1., 2.],
                            '25 to 45':         [1., 0., 1.],
                            'Greater than 45':  [2., 1., 0.]},
                            index=['Less than 25', '25 to 45', 'Greater than 45'])
    distort['c_charge_degree'] = pd.DataFrame(
                            {'M':   [0., 2.],
                            'F':    [1., 0.]},
                            index=['M', 'F'])
    distort['priors_count'] = pd.DataFrame(
                            {'0':           [0., 1., 2.],
                            '1 to 3':       [1., 0., 1.],
                            'More than 3':  [2., 1., 0.]},
                            index=['0', '1 to 3', 'More than 3'])
    distort['sex'] = pd.DataFrame(
                        {0.0:    [0., 2.],
                         1.0:    [2., 0.]},
                         index=[0.0, 1.0])
    distort['race'] = pd.DataFrame(
                        {0.0:    [0., 2.],
                         1.0:    [2., 0.]},
                         index=[0.0, 1.0])

    total_cost = 0.0
    for k in vold:
        if k in vnew:
            total_cost += distort[k].loc[vnew[k], vold[k]]

    return total_cost



def get_distortion_generic(vold, vnew, label_name='Target'):
    """
    Função de distorção genérica que aplica um custo de 1.0 se o rótulo da classe
    for alterado, e um custo de 0.0 caso contrário.

    Esta é uma implementação da "perda 0-1" na utilidade, onde a principal
    perda de utilidade é considerada a inversão da predição original.

    Args:
        vold (dict): Dicionário {coluna: valor_antigo}
        vnew (dict): Dicionário {coluna: valor_novo}
        label_name (str): O nome da coluna do rótulo/target.

    Returns:
        float: O custo da distorção (0.0 ou 1.0).
    """
    # Usamos .get() para evitar erro se a chave não existir em um dos dicts
    if vold.get(label_name) != vnew.get(label_name):
        # Penaliza com custo 1 se o rótulo da classe mudou
        return 1.0
    else:
        # Nenhum custo se o rótulo permaneceu o mesmo
        return 0.0
    
    
def get_distortion_custom(vold, vnew, protected_attribute_name, label_name):
    """
    Função de distorção customizada que aplica uma hierarquia de custos
    para as alterações feitas pelo OptimPreproc.

    Hierarquia de Custos:
    1. Custo muito alto se o atributo protegido for alterado.
    2. Custo de 1.0 se o rótulo da classe (Target) for alterado.
    3. Custo de 0.1 para cada outra feature que for alterada.

    Args:
        vold (dict): Dicionário {coluna: valor_antigo}
        vnew (dict): Dicionário {coluna: valor_novo}
        protected_attribute_name (str): O nome da coluna do atributo protegido.
        label_name (str): O nome da coluna do rótulo/target.

    Returns:
        float: O custo total da distorção.
    """
    # 1. Penalidade máxima se o atributo protegido for alterado
    if vold.get(protected_attribute_name) != vnew.get(protected_attribute_name):
        return 1e6  # Retorna um número muito grande para tornar essa mudança proibitiva

    total_cost = 0.0

    # 2. Penalidade se o rótulo da classe for alterado
    if vold.get(label_name) != vnew.get(label_name):
        total_cost += 1.0

    # 3. Penalidade pequena para cada outra feature alterada
    for key in vold:
        # Pula a verificação para as chaves já tratadas
        if key not in [protected_attribute_name, label_name]:
            if vold.get(key) != vnew.get(key):
                total_cost += 0.1 # Custo pequeno por feature alterada

    return total_cost
