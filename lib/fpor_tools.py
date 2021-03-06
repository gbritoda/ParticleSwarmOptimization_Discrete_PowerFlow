import numpy as np
import pandapower as pp
import pandapower.networks

def sup_discrete_values(vetor_x, lista_discretos):
    '''
    Função que retorna o valor discreto superior de 'lista_discretos' mais próximo de todos os valores x de 'vetor_x' 

    Inputs:
        -> vetor_x = vetor (numpy array) contendo os valores que deseja-se obter o número discreto mais próximos
        -> lista_discretos = lista (python list) que contém o conjunto de valores discretos que cada variável x admite
    
    Ouputs:
        -> x_sup = vetor (numpy array) contendo os valores discretos superiores de 'lista_discretos' mais próximo 
           dos valores de 'vetor_x'
    '''
    #Vetor de saída da função. Possui o mesmo formato (shape) que vetor_x
    x_sup = np.zeros(vetor_x.shape)
    
    #Cópia de 'vetor_x'. Esta cópia é feita para evitar erros de alocamento dinâmico de memória.
    vetor = np.copy(vetor_x)
    
    #Garante que a lista seja uma array numpy e armazena o resultado na variável 'lista'
    lista = np.asarray(lista_discretos, dtype = np.float32)
    
    '''
    Garante que os valores de 'vetor_x' estejam dentro dos limites de 'lista discretos' por um pequeno fator de 10^-3.
    Caso contrário, a função numpy.searchsorted descrita a frente resultará em erro.
    '''
    vetor = vetor - 1e-3
    np.clip(a = vetor, a_min = lista[0] + 1e-3, a_max = lista[-1] - 1e-3, out = vetor)
    
    '''
    Utilizando a função numpy.searchsorted() para buscar os índices de 'lista_discretos' que correspondem aos valores
    discretos superiores aos valores de 'vetor_x'
    '''
    indices = np.searchsorted(a=lista, v = vetor, side='right')
    
    #Armazena os valores de 'lista_discretos' cujos índices correspondem aos discretos superiores de 'vetor_x'
    x_sup = np.take(lista, indices)
    
    #Deleta as variáveis locais
    del vetor, lista, indices
    
    return x_sup

def inf_discrete_values(vetor_x, lista_discretos):
    '''
    Função que retorna o valor discreto inferior de 'lista_discretos' mais próximo de todos os valores x de 'vetor_x' 
    
    Inputs:
        -> vetor_x = vetor (numpy array) contendo os valores que deseja-se obter o número discreto mais próximo
        -> lista_discretos = lista (python list) que contém o conjunto de valores discretos que cada variável x admite
    
    Ouputs:
        -> x_inf = vetor (numpy array) contendo os valores discretos inferiores de 'lista_discretos' mais próximos 
           dos valores de 'vetor_x'
    '''
    
    #Vetor de saída da função. Possui o mesmo formato (shape) que vetor_x
    x_inf = np.zeros(vetor_x.shape)
    
    #Cópia de 'vetor_x'. Esta cópia é feita para evitar erros de alocamento dinâmico de memória.
    vetor = np.copy(vetor_x)
    
    #Garante que a lista seja uma array numpy e salva o resultado na variável local 'lista'
    lista = np.asarray(lista_discretos, dtype = np.float32)
    
    '''
    Garante que os valores de 'vetor_x' estejam dentro dos limites de 'lista discretos' por um pequeno fator de 10^-3.
    Caso contrário, a função numpy.searchsorted descrita a frente resultará em erro. Salva o resultado de numpy.clip
    na variável local 'vetor'
    '''
    vetor = vetor - 1e-3
    np.clip(a = vetor, a_min = lista_discretos[0] + 1e-3, a_max = lista_discretos[-1] - 1e-3, out = vetor)
    
    '''
    Utilizando a função numpy.searchsorted() para buscar os índices de 'lista_discretos' que correspondem aos valores
    discretos inferiores aos valores de 'vetor_x'
    '''
    indices = np.searchsorted(a=lista, v = vetor, side='left') - 1
    
    #Armazena os valores de 'lista_discretos' cujos índices correspondem aos discretos superiores de 'vetor_x'
    x_inf = np.take(lista, indices)
    
    #Deleta as variáveis locais
    del vetor, lista, indices
    
    return x_inf

def network_set(net):
    
    """
    Esta funcao organiza a net obtida do PandaPower de modo que ela possa ser mais facilmente utilizada pelo algoritmo.
    Suas funcionalidades são:
        
        -> Ordenar os parâmetros da net (v_bus, tap, shunt, etc) por índices;
        -> Obter os transformadores com controle de tap;
        -> Gerar o vetor com os valores dos taps dos transformadores;
        -> Gerar as matrizes com os valores discretos para os shunts de cada sistema;
        -> Gerar as matrizes de mascaramento com as probabilidades de escolha dos shunts para cada sistema;
        -> Gerar o primeiro agente de busca (que contém as variáveis do ponto de operação do sistema);
        -> Obter as condutâncias das linhas.
        
    Input:
        -> net
        
    Output:
        
        -> net gerenciada (não devolvida, salva diretamente na variável net);
        -> vetor de condutâncias da net: G_net;
        -> matriz das linhas de transmissao: linhas;
        -> vetor contendo os valores discretos para os taps: valores_taps;
        -> matrizes contendo os valores discretos para os shunts: valores_shunts;
        -> matrizes de mascaramento dos shunts: mask_shunts.
    """
    
    #Ordenar os índices da net
    net.bus = net.bus.sort_index()
    net.res_bus = net.res_bus.sort_index()
    net.gen = net.gen.sort_index()
    net.line = net.line.sort_index()
    net.shunt = net.shunt.sort_index()
    net.trafo = net.trafo.sort_index()
    
    #Algumas nets são inicializadas com taps negativos
    net.trafo.tap_pos = np.abs(net.trafo.tap_pos)

    #É preciso ordenar os taps para remover os transformadores sem controle de tap
    net.trafo = net.trafo.sort_values('tap_pos')

    #num_trafo_controlado: variavel para armazenar o número de trafos com controle de tap
    num_trafo_controlado = net.trafo.tap_pos.count()
    
    #num_barras : variavel utilizada para salvar o numero de barras do sistema
    num_barras = net.bus.name.count()
    
    #num_shunt: variavel para armazenar o numero de shunts do sistema
    num_shunt = net.shunt.in_service.count()
    
    #num_gen: variavel para armazenar o número de barras geradoras do sistema
    num_gen = net.gen.in_service.count()
    
    '''
    Cria as varíaveis globais nb, nt, ns, ng para facilitar o uso desses parâmetros em outros funções
    
    '''
    global nb, nt, ns, ng
    nb, nt, ns, ng = num_barras, num_trafo_controlado, num_shunt, num_gen
    
    '''
    Muda os valores máximos e mínimos permitidos das tensões das barras dos sistemas de 118 e 300 barras:
        min_vm_pu: 0.94 -> 0.90
        max_vm_pu: 1.06 -> 1.10
    '''
    if nb == 118 or nb == 300:
        net.bus.min_vm_pu = 0.90
        net.bus.max_vm_pu = 1.10
    
    #Dicionário que contem os valores dos shunts para cada sistema IEEE
    valores_shunts = {"14": np.array([
                            [0.0, 0.19, 0.34, 0.39]
                            ]),
                      
                      "30": np.array([
                            [0.0, 0.19, 0.34, 0.39],
                            [0.0, 0.0, 0.05, 0.09]
                            ]),
                      
                      "57": np.array([
                            [0.0, 0.12, 0.22, 0.27], 
                            [0.0, 0.04, 0.07, 0.09], 
                            [0.0, 0.0, 0.10, 0.165]
                            ]),
                      
                      "118": np.array([
                              [-0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.06, 0.07, 0.13, 0.14, 0.2],
                              [-0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.15],
                              [0.0, 0.0, 0.0, 0.08, 0.12, 0.2],
                              [0.0, 0.0, 0.0, 0.0, 0.1, 0.2],
                              [0.0, 0.0, 0.0, 0.0, 0.1, 0.2],
                              [0.0, 0.0, 0.0, 0.0, 0.1, 0.2],
                              [0.0, 0.0, 0.0, 0.0, 0.1, 0.2],
                              [0.0, 0.06, 0.07, 0.13, 0.14, 0.2],
                              [0.0, 0.06, 0.07, 0.13, 0.14, 0.2]
                              ]),
                      
                      "300": np.array([
                              [0.0, 2.0, 3.5, 4.5],
                              [0.0, 0.25, 0.44, 0.59],
                              [0.0, 0.19, 0.34, 0.39],
                              [-4.5, 0.0, 0.0, 0.0],
                              [-4.5, 0.0, 0.0, 0.0],
                              [0.0, 0.25, 0.44, 0.59],
                              [0.0, 0.25, 0.44, 0.59],
                              [-2.5, 0.0, 0.0, 0.0],
                              [-4.5, 0.0, 0.0, 0.0],
                              [-4.5, 0.0, 0.0, 0.0],
                              [-1.5, 0.0, 0.0, 0.0],
                              [0.0, 0.25, 0.44, 0.59],
                              [0.0, 0.0, 0.0, 0.15],
                              [0.0, 0.0, 0.0, 0.15]
                              ])
                      }
    
    mask_shunts = {
        '14': np.array([
              [0.25, 0.25, 0.25, 0.25]
        ]),
        '30': np.array([
              [0.25, 0.25, 0.25, 0.25],
              [0.0, 1./3., 1./3., 1./3.]
              ]),
        '57': np.array([
              [0.25, 0.25, 0.25, 0.25], 
              [0.25, 0.25, 0.25, 0.25], 
              [0.0, 1./3., 1./3., 1./3.]
              ]),
        '118': np.array([
              [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
              [1./6., 1./6., 1./6., 1./6., 1./6., 1./6.],
              [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
              [0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
              [0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
              [0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
              [0.0, 0.0, 0.25, 0.25, 0.25, 0.25],
              [0.0, 0.0, 0.0, 1./3., 1./3., 1./3.],
              [0.0, 0.0, 0.0, 1./3., 1./3., 1./3.],
              [0.0, 0.0, 0.0, 1./3., 1./3., 1./3.],
              [0.0, 0.0, 0.0, 1./3., 1./3., 1./3.],
              [1./6., 1./6., 1./6., 1./6., 1./6., 1./6.],
              [1./6., 1./6., 1./6., 1./6., 1./6., 1./6.]
              ]),
        '300': np.array([
              [0.25, 0.25, 0.25, 0.25],
              [0.25, 0.25, 0.25, 0.25],
              [0.25, 0.25, 0.25, 0.25],
              [0.5, 0.5, 0.0, 0.0],
              [0.5, 0.5, 0.0, 0.0],
              [0.25, 0.25, 0.25, 0.25],
              [0.25, 0.25, 0.25, 0.25],
              [0.5, 0.5, 0.0, 0.0],
              [0.5, 0.5, 0.0, 0.0],
              [0.5, 0.5, 0.0, 0.0],
              [0.5, 0.5, 0.0, 0.0],
              [0.25, 0.25, 0.25, 0.25],
              [0.0, 0.0, 0.5, 0.5],
              [0.0, 0.0, 0.5, 0.5]
              ])
    }
    
    
    #Vetor que contém os valores discretos dos taps: entre 0.9 e 1.1 com passo = tap_step
    global tap_step
    tap_step = 0.00625
    valores_taps = np.arange(start = 0.9, stop = 1.1, step = tap_step)
    
    """
    Matriz contendo as linhas de transmissão da net:
        -> linhas[0] = vetor com as barras de ínicio;
        -> linhas[1] = vetor com as barras de término;
        -> linhas[2] = vetor com as resistências em pu (r_pu) das linhas;
        -> linhas[3] = vetor com as reatâncias em pu (x_pu) das linhas.
    r_pu = r_ohm/z_base
    x_pu = x_ohm/z_base
    g = r_pu/(r_pu^2 + x_pu^2)
    """
    
    linhas = np.zeros((4, net.line.index[-1]+1))
    linhas[0] = net.line.from_bus.to_numpy()
    linhas[1] = net.line.to_bus.to_numpy()
    v_temp = net.bus.vn_kv.to_numpy()
    z_base = np.power(np.multiply(v_temp,1000), 2)/100e6
    for i in range(net.line.index[-1]+1):
        linhas[2][i] = net.line.r_ohm_per_km[i]/z_base[int(linhas[0][i])]
        linhas[3][i] = net.line.x_ohm_per_km[i]/z_base[int(linhas[0][i])]
    del v_temp, z_base
    
    #Vetor G_net com as condutâncias das linhas de transmissão
    G_net = np.zeros((1, net.line.index[-1]+1))
    G_net = np.array([np.divide(linhas[2], np.power(linhas[2],2)+np.power(linhas[3],2))])
    
    #Matriz de condutância nodal da net. É equivalente à parte real da matriz de admintância nodal do sistema
    matriz_G = np.zeros((num_barras,num_barras))
    matriz_G[linhas[0].astype(np.int), linhas[1].astype(np.int)] = G_net 
    
    """
    tap_pu = (tap_pos + tap_neutral)*tap_step_percent/100 (equação fornecida pelo PandaPower)
    shunt_pu = -100*shunt (equação fornecida pelo PandaPower)
    
    As variáveis v_temp, taps_temp e shunt_temp são utilizadas para receber os valores de tensão, tap e shunt da net
    """
    
    net_params = {"lines": linhas,
                       "shunt_values": valores_shunts,
                       "tap_values": valores_taps,
                       "shunt_masks": mask_shunts,
                       "G_matrix": matriz_G,
                       "n_bus": nb,
                       "n_taps": nt,
                       "n_shunt": ns,
                       "n_gen": ng,
                       "tap_step": tap_step}

    return net_params

def get_upper_and_lower_bounds_from_net(net, net_params):
    #TBD Description
    nb = net_params['n_bus']
    ng = net_params['n_gen']
    nt = net_params['n_taps']
    # ns = net_params['n_shunt']

    shunt_values = net_params['shunt_values'][str(nb)]
    v_max = net.bus.max_vm_pu.to_numpy(dtype = np.float32)[:ng]
    v_min = net.bus.min_vm_pu.to_numpy(dtype = np.float32)[:ng]
    tap_max = np.repeat(net_params['tap_values'][-1], nt)
    tap_min = np.repeat(net_params['tap_values'][0], nt)
    shunt_max = shunt_values[:,-1]
    shunt_min = shunt_values[:, 0]

    upper_bounds = np.squeeze(np.array([np.concatenate((v_max, tap_max, shunt_max),axis=0)]))
    lower_bounds = np.squeeze(np.array([np.concatenate((v_min, tap_min, shunt_min),axis=0)]))
    
    return upper_bounds, lower_bounds

def get_variables_number_from_net(net, net_params):

    # independent variables
    # Voltages
    v_temp = net.gen.vm_pu.to_numpy(dtype=np.float32)

    #  tap_pu = (tap_pos + tap_neutral)*tap_step_percent/100 (PandaPower)
    taps_temp = 1 + ((net.trafo.tap_pos.to_numpy(dtype = np.float32)[0:nt] +\
                        net.trafo.tap_neutral.to_numpy(dtype = np.float32)[0:nt]) *\
                        (net.trafo.tap_step_percent.to_numpy(dtype = np.float32)[0:nt]/100))

    #  shunt_pu = -100*shunt (PandaPower)
    shunt_temp = -net.shunt.q_mvar.to_numpy(dtype = np.float32)/100

    # Particles will be in the following format:
    x = np.squeeze(np.array([np.concatenate((v_temp, taps_temp, shunt_temp),axis=0)]))

    # nVar = len(x)
    return len(x)


def fopt_and_penalty(net, net_params, n_threshold=1e-12):
    """
    E função  calcula a função objetivo para o problema de FPOR e também calcula
    a penalização das tensões que ultrapassam o limite superior ou ficam abaixo do limite
    inferior para cada agente de busca.
    
    Esta função não é vetorizada para toda a alcateia
    
    A função objetivo deste problema dá as perdas de potência ativa no SEP.
    
    f = sum g_km * [v_k^2 + v_m^2 - 2*v_k*v_m*cos(theta_k - theta_m)]
    
    A penalização das tensões é:
        
    pen_v = sum(v - v_lim)^2
        v_lim = v_lim_sup, se v > v_lim_sup
        v_lim = v, se v_lim_inf < v < v_lim_sup
        v_lim = v_lim_inf, se v < v_lim_inf
        
    Inputs:
        -> net = sistema elétrico de testes
        -> G_matrix = matriz de condutância nodal do sistema
        -> v_lim_sup = vetor contendo os limites superiores de tensão nas barras do sistema
        -> v_lim_inf = vetor contendo os limites inferiores de tensão nas barras do sistema
    Outputs:
        -> f = função objetivo do problema de FPOR
        -> pen_v = penalidade de violação dos limites de tensão das barras
    
    """
    
    G_matrix = net_params["G_matrix"]

    v_lim_sup = net.bus.max_vm_pu.to_numpy(dtype = np.float32)
    v_lim_inf = net.bus.min_vm_pu.to_numpy(dtype = np.float32)

    v_k = np.array([net.res_bus.vm_pu.to_numpy(dtype = np.float64)])
    v_m = v_k.T
    
    theta_k = np.radians(np.array([net.res_bus.va_degree.to_numpy(dtype = np.float64)]))
    theta_m = theta_k.T
    
    #Calculo da função objetivo
    f = np.power(v_k,2) + np.power(v_m,2) - 2*np.multiply(np.multiply(v_k,v_m),np.cos(theta_k-theta_m))
    f = np.multiply(G_matrix, f)
    f = np.squeeze(np.sum(np.array([f[np.nonzero(f)]])))
    
    #Calculo da penalidade das tensões
    
    #Violação do limite superior (v - v_lim), v > v_lim_sup
    v_up = v_k - v_lim_sup
    v_up = v_up[np.greater(v_up, 0.0)]

    
    #Violação do limite inferior (v-v_lim), v < v_lim_inf
    v_down = v_k - v_lim_inf
    v_down = v_down[np.less(v_down, 0.0)]
    
    #Threshold
    up_thr = np.less_equal(v_up, n_threshold)
    down_thr = np.less_equal(v_down, n_threshold)
    
    v_up[up_thr] = 0.0
    v_down[down_thr] = 0.0
    
    # Boundaries penalty
    pen_v = np.squeeze(np.sum(np.square(v_up)) + np.sum(np.square(v_down)))
    if pen_v <= n_threshold:
        pen_v = 0.0

    return f, pen_v

def senoidal_penalty_taps(particle, net_params, n_threshold=1e-12, debug=False):
    '''
    TBD Description
    Esta função retorna a penalidade senoidal sobre os taps dos transformadores para toda a alcateia.
    Dado um tap t e um passo discreto s, a função penalidade senoidal é dada por:
        pen_sen_tap = sum {sen^2(t*pi/s)}
    
    Inputs:
        -> particle
        -> net_params
        -> n_threshold
    Outputs:
        -> pen_taps
    '''
    ng = net_params['n_gen']
    nt = net_params['n_taps']
    taps = particle[ng:ng+nt]

    pen_taps = np.square(np.sin(taps*np.pi/tap_step))

    threshold = np.less_equal(pen_taps, n_threshold)
    pen_taps[threshold] = 0.0
    if debug:
        print("pen_taps vector: {}".format(pen_taps))
    pen_taps = np.sum(pen_taps, axis=0)

    del taps
    
    return pen_taps

# def senoidal_penalty_shunt(particle, net_params, n_threshold=1e-12):
#     '''
#     TBD Description
#     Esta função retorna a penalidade senoidal sobre os shunts de toda a 
#     alcateia.
    
#     Seja 'conjunto' a lista de valores discretos que o shunt admite: 
#         conjunto = [b_1, b_2, b_3, b_4].
#     Seja b um shunt
    
#     A função pen_sen_shunt deve ser nula caso 'b' pertença a 'conjunto' e 
#     maior que 0 caso contrário.
    
#     Define-se a variável a função pen_sen_shunt para o caso de um único shunt b como:
        
#         pen_sen_shunt = sen[ pi * (b /(b_sup - b_inf)) + alfa ]
    
#     Onde:
#         - b_sup: é o valor discreto superior mais próximo de 'b'
#         - b_inf: é o valor discreto inferior mais próximo de 'b'
#         - alfa: é uma variável escolhida para que pen_sen_shunt = 0 caso 'b' pertença a 'conjunto'
    
#     Alfa é dada por:
        
#         alfa = pi*[ ceil{b_inf/(b_sup - b_inf)} - b_inf/(b_sup - b_inf)]
        
#         *ceil(x) é o valor de x arredondado para o inteiro superior mais próximo
    
#     Inputs:
#         -> alcateia 
#         -> conjunto_shunts = conjunto de valores que cada shunt de 'alcateia' pode admitir
        
#     Outputs:
#         -> pen_shunts: um vetor cuja forma é (ns, n_lobos) contendo as penalizações referentes aos shunts para toda a alcateia
    
#     '''

#     #TBD Description
#     nb = net_params['n_bus']
#     ng = net_params['n_gen']
#     # nt = net_params['n_taps']
#     ns = net_params['n_shunt']

#     shunts = np.array(particle[ng+nt:ng+nt+ns])
    
#     #Variáveis temporárias para armazenar b_inf's e b_sup's, obtidos pelas funções auxiliares descritas no ínicio do código
#     shunts_sup = np.zeros(shape = shunts.shape)
#     shunts_inf = np.zeros(shape = shunts.shape)

#     shunts = net_params["shunt_values"][str(nb)]
    
#     for idx, shunt_value in enumerate(shunts):
#         shunts_sup[idx] = sup_discrete_values(shunts[idx], shunt_value)
#         shunts_inf[idx] = inf_discrete_values(shunts[idx], shunt_value)

#     d = np.abs(shunts_sup - shunts_inf)

#     alfa = np.pi * (np.ceil(np.abs(shunts_inf)/d) - np.abs(shunts_inf)/d)
    
#     pen_shunts = np.sin(alfa + np.pi*(shunts/d))
#     pen_shunts = np.square(pen_shunts)
#     if not DEBUG:
#         pen_shunts = np.sum(pen_shunts, axis = 0, keepdims = True)
#     threshold = np.less_equal(pen_shunts, 1e-5)
#     pen_shunts[threshold] = 0.0
    
#     return pen_shunts

def polinomial_penalty_shunts(particle, net_params, n_threshold=1e-12):
    # TBD Description
    
    nb = net_params['n_bus']
    ng = net_params['n_gen']
    ns = net_params['n_shunt']
    
    shunts = np.array(particle[ng+nt:ng+nt+ns])
    shunt_set = np.array(net_params['shunt_values'][str(nb)])

    # If value is less than threshold, we consider them zero
    sh_vector = shunts - shunt_set.T
    threshold = np.less_equal(sh_vector, n_threshold)
    sh_vector[threshold] = 0.0

    pen_shunts = np.power(np.prod((shunts-shunt_set.T), axis=0),2).sum()

    return pen_shunts



def run_power_flow(particle, net, net_params, ng, nt, ns):
    '''
    TBD Description
    Inputs:
        -> alcateia
        -> net
        -> conjunto_shunts
    Outputs:
        -> alcateia
    '''
    
    '''
    Variável que armazena o número de variáveis do problema.
    ng = número de barras geradoras do sistema;
    nt = número de transformadores com controle de tap do sistema;
    ns = número de susceptâncias shunt do sistema.
    '''

    v_gen = particle[:ng]
    taps = particle[ng:ng+nt]
    shunts = particle[ng+nt:ng+nt+ns]

    #Inserindo as tensões das barras de geração na net
    net.gen.vm_pu = v_gen
        
    #Inserindo os taps dos transformadores
    '''
    Os taps dos transformadores devem ser inseridos como valores de posição, 
    e não como seu valor em pu. Para converter de pu para posição é utilizada a seguinte equação:
    
        tap_pos = [(tap_pu - 1)*100]/tap_step_percent] + tap_neutral

    O valor tap_mid_pos é 0 no sistema de 14 barras
    '''
    net.trafo.tap_pos[:nt] = net.trafo.tap_neutral[:nt] + ((taps - 1.0)*(100/net.trafo.tap_step_percent[:nt]))
        
    #Inserindo as susceptâncias shunt
    """
    A unidade de susceptância shunt no pandapower é MVAr e seu sinal é negativo. 
    Para transformar de pu para MVAr negativo basta multiplicar por -100
    """
    net.shunt.q_mvar = shunts*(-100)
        
    #Soluciona o fluxo de carga utilizando o algoritmo Newton-Raphson
    pp.runpp(net, algorithm = 'nr', numba = True, init = 'results', tolerance_mva = 1e-5)
        
    #Recebendo os valores das tensões das barras, taps e shunts e armazenando na particula
    v_gen = net.res_gen.vm_pu.to_numpy(dtype = np.float32)

    #Recebendo a posição dos taps e convertendo pra pu
    taps = 1 + ((net.trafo.tap_pos[:nt] - net.trafo.tap_neutral[:nt])*(net.trafo.tap_step_percent[:nt]/100))
        
    #Recebendo o valor da susceptância shunt e convertendo para pu
    shunts = net.res_shunt.q_mvar.to_numpy(dtype = np.float32)/(-100) 
        
    #Atualizando 
    particle[:ng] = v_gen
    particle[ng:ng+nt] = taps
    particle[ng+nt:ng+nt+ns] = shunts

    return particle

def debug_fitness_function(particle, net, net_params, test_params, printout=True):
    #TBD Description
    # nb = net_params['n_bus']
    ng = net_params['n_gen']
    ns = net_params['n_shunt']
    nt = net_params['n_taps']
    lambd_volt = test_params['lambda_volt']
    lambd_tap = test_params['lambda_tap']
    lambd_shunt = test_params['lambda_shunt']
    tap_thr = test_params['tap_threshold']
    sh_thr = test_params['shunt_threshold']

    # v_gen = particle[:ng]
    # taps = particle[ng:ng+nt]
    # shunts = particle[ng+nt:ng+nt+ns]

    particle = run_power_flow(particle, net, net_params, ng, nt, ns)
    #fopt and boundaries penalty
    f, pen_v = fopt_and_penalty(net, net_params)
    tap_pen = senoidal_penalty_taps(particle, net_params, n_threshold=tap_thr)
    shunt_pen = polinomial_penalty_shunts(particle, net_params, n_threshold=sh_thr)

    if printout:
        print('Fopt:\t\t{}\npen_v:\t\t{}\ntap_pen:\t{}\nshunt_pen:\t{}\nlambda_volt:\t{}\nlambda_tap:\t{}\nlambda_shunt:\t{}\n'.format(
            f, pen_v, tap_pen, shunt_pen, lambd_volt, lambd_tap, lambd_shunt))
    ret = {
        'f':f,
        'pen_v':pen_v,
        'tap_pen':tap_pen,
        'shunt_pen':shunt_pen,
        'lambda_volt':lambd_volt,
        'lambda_tap':lambd_tap,
        'lambda_shunt':lambd_shunt,
        'particle':particle
    }
    return ret
