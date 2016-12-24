def process_df(sim, name, lower, upper, step, states):
    import numpy as np
    from pymbar.timeseries import statisticalInefficiency

    # get data for every `step`
    df = sim.data.retrieve(name)

    df = df.loc[lower:upper]

    # drop any rows that have missing values
    df = df.dropna()

    # subsample according to statistical inefficiency after equilibration detection
    # we do this after slicing by lower/upper to simulate
    # what we'd get with only this data available
    #statinef  = statisticalInefficiency(df[df.columns[sim.categories['state']]])

    # we round up
    #statinef = int(np.rint(statinef))

    #df = df.to_delayed()[0]
    
    # subsample
    df = df.iloc[::step]

    # extract only columns that have the corresponding sim present        
    df = df[df.columns[states]]

    # subsample according to statistical inefficiency and equilibrium detection
    #df = df.iloc[::statinef]
    
    return df
