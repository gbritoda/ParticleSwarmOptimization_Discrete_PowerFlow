import numpy as np
import pandapower as pp
import pandapower.networks

def network_set(net):
    
    """
    Esta funcao organiza a rede obtida do PandaPower de modo que ela possa ser mais facilmente utilizada pelo algoritmo.
    Suas funcionalidades são:
        
        -> Ordenar os parâmetros da rede (v_bus, tap, shunt, etc) por índices;
        -> Obter os transformadores com controle de tap;
        -> Gerar o vetor com os valores dos taps dos transformadores;
        -> Gerar as matrizes com os valores discretos para os shunts de cada sistema;
        -> Gerar as matrizes de mascaramento com as probabilidades de escolha dos shunts para cada sistema;
        -> Gerar o primeiro agente de busca (que contém as variáveis do ponto de operação do sistema);
        -> Obter as condutâncias das linhas.
        
    Input:
        -> rede
        
    Output:
        
        -> rede gerenciada (não devolvida, salva diretamente na variável rede);
        -> primeiro agente de buscas: lobo_1;
        -> vetor de condutâncias da rede: G_rede;
        -> matriz das linhas de transmissao: linhas;
        -> vetor contendo os valores discretos para os taps: valores_taps;
        -> matrizes contendo os valores discretos para os shunts: valores_shunts;
        -> matrizes de mascaramento dos shunts: mask_shunts.
    """
    
    #Ordenar os índices da rede
    net.bus = net.bus.sort_index()
    net.res_bus = net.res_bus.sort_index()
    net.gen = net.gen.sort_index()
    net.line = net.line.sort_index()
    net.shunt = net.shunt.sort_index()
    net.trafo = net.trafo.sort_index()
    
    #Algumas redes são inicializadas com taps negativos
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
    #Precisa ser um tensor de rank 1 para que a alcateia possa ser inicializada
    global tap_step
    tap_step = 0.00625
    valores_taps = np.arange(start = 0.9, stop = 1.1, step = tap_step)
    
    """
    Matriz contendo as linhas de transmissão da rede:
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
    
    #Vetor G_rede com as condutâncias das linhas de transmissão
    G_rede = np.zeros((1, net.line.index[-1]+1))
    G_rede = np.array([np.divide(linhas[2], np.power(linhas[2],2)+np.power(linhas[3],2))])
    
    #Matriz de condutância nodal da net. É equivalente à parte real da matriz de admintância nodal do sistema
    matriz_G = np.zeros((num_barras,num_barras))
    matriz_G[linhas[0].astype(np.int), linhas[1].astype(np.int)] = G_rede 
    
    """
    O primeiro lobo (agente de busca) será inicializado com os valores de operação da rede fornecidos
    pelo PandaPower: vetor lobo_1.
    
    tap_pu = (tap_pos + tap_neutral)*tap_step_percent/100 (equação fornecida pelo PandaPower)
    shunt_pu = -100*shunt (equação fornecida pelo PandaPower)
    
    As variáveis v_temp, taps_temp e shunt_temp são utilizadas para receber os valores de tensão, tap e shunt da rede
    e armazenar no vetor lobo_1
    """
    
    # v_temp = net.gen.vm_pu.to_numpy(dtype = np.float32)
    
    # taps_temp = 1 + ((net.trafo.tap_pos.to_numpy(dtype = np.float32)[0:num_trafo_controlado] +\
    #                   net.trafo.tap_neutral.to_numpy(dtype = np.float32)[0:num_trafo_controlado]) *\
    #                  (net.trafo.tap_step_percent.to_numpy(dtype = np.float32)[0:num_trafo_controlado]/100))
        
    # shunt_temp = -net.shunt.q_mvar.to_numpy(dtype = np.float32)/100
    
    # del v_temp, taps_temp, shunt_temp

    #     global nb, nt, ns, ng
    # nb, nt, ns, ng = num_barras, num_trafo_controlado, num_shunt, num_gen

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

def fopt_and_penalty(net, net_params):
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
        -> rede = sistema elétrico de testes
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
    f=np.squeeze(np.sum(np.array([f[np.nonzero(f)]])))
    
    #Calculo da penalidade das tensões
    
    #Violação do limite superior (v - v_lim), v > v_lim_sup
    v_up = v_k - v_lim_sup
    v_up = v_up[np.greater(v_up, 0.0)]
    
    #Violação do limite inferior (v-v_lim), v < v_lim_inf
    v_down = v_k - v_lim_inf
    v_down = v_down[np.less(v_down, 0.0)]
    
    pen_v = np.squeeze(np.sum(np.square(v_up)) + np.sum(np.square(v_down)))

    return f, pen_v