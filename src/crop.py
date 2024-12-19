

def crop_statements_until_t(df, t):
    
    df = df[df.time<t]
    idspoliticos = df.Id_politico.unique()
    tau = []
    
    for i in idspoliticos:
        taui = df.isFavorable[df.Id_politico == i]
        tau.append([taui.tolist(),i])    #inclui iD politico
    
    return tau

def crop_all_statements(df):

    idspoliticos = df.Id_politico.unique()
    
    tau = []
    
    for i in idspoliticos:
        taui = df.isFavorable[df.Id_politico == i]
        tau.append([taui.tolist()])    # nao inclui iD politico
    
    return tau

def crop_all_statements_per_politician(df, id_politico ):

    taui = df.isFavorable[df.Id_politico == id_politico]
    
    return taui.tolist()

def crop_statements_from_t0_to_t(df, t0, t):

    _df = df[df.time>t0]
    _df = _df[_df.time<t]
    idspoliticos = _df.Id_politico.unique()
    tau = []
    
    for i in idspoliticos:

        taui = _df.isFavorable[_df.Id_politico == i]
        tau.append([taui.tolist(), i])    #inclui iD politico
    
    return tau

def crop_statements_until_t_by_politician(df, t, id_politician):
    
    df = df[df.time<t]

    taui = df.isFavorable[df.Id_politico == id_politician]
    tau = taui.tolist()   
    
    return tau